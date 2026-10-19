"""Synthesise signing-hand landmark sequences from text with the released OpenFS diffusion generator.

Research-only, and it has to stay that way. The released generator and the released conditioner
carry the OpenFS Non-Commercial Research License: "Commercial use of the source code, model
checkpoints, or derivative works is strictly prohibited" (data/external/openfs/source/LICENSE,
data/openfs-checkpoints/generator_df_fsboard/PROVENANCE.json). Sequences sampled by a released
checkpoint are a derivative work of it, so nothing this script writes may train anything the app
ships. Every generation directory records the generator sha256, the conditioner sha256, the word
list hashes, the trial count and the seed, and refuses to run when those are not pinned.

What conditions the sampler is the per-frame character label sequence, not the released
recognizer: lib.models.generator_df.EncoderTransformer embeds frame_label plus frame_idx, and the
recognizer named in the checkpoint's .hydra/config.yaml is loaded only to transcribe each generated
clip. That transcription is this script's automatic keep/drop signal, so it is a model-derived
quality filter, never a human check, and any kept pack inherits the released recognizer's biases.
The transcription rule is --quality-rule: exact (the whole transcription is the word) or prefix (it
starts with the word). The released decoder is greedy at max_len=32 and frequently never emits EOS
on generated poses, so it appends hallucinated characters after a word it read correctly; exact
keeps only 4% of clips overall and almost nothing on single letters, which would hollow out the
J/Z experiment. Both rates are always reported, whichever rule kept the pack.

The released code is imported, not reimplemented: lib.models.diffusion.Diffusion.sampling (ddim,
50 steps of a T=1000 cosine schedule), lib.models.generator_df.EncoderTransformer,
lib.datasets.fingerspelling.WordDataset with use_frame_label_clean=true and
lib.utils.dataloader.paired_collate_word_dataset_fn, replaying generate_data_df.py with our own
per-clip records and provenance. The word-level frame labels WordDataset builds are the same
label layout our own packer aligns to in scripts/prepare_openfs_training.py, which is what makes
them usable as per-frame training targets.

The VPS venv has no tqdm, scipy or scikit-learn and upstream imports all three at module level.
upstream_dependency_stubs() replaces them and nothing else: tqdm only wraps an iterator,
scipy.ndimage.zoom is training-time pose resampling and sklearn.preprocessing only serves the CTC
CSV helpers, none of them on the generation path. Every stubbed callable raises the moment upstream
code calls it, and which stubs were installed is recorded in the provenance file.

Pilot (50 FSboard train words, 3 trials, a couple of minutes on a V100-SXM3-32GB):

  ASL_OPENFS_SOURCE=$PWD/data/external/openfs/source .venv/bin/python scripts/generate_openfs_synthetic.py --source data/external/openfs/source --generator data/openfs-checkpoints/generator_df_fsboard --conditioner data/openfs-checkpoints/recognizer_fsboard --generator-sha256 4733de5601d5ee25162b0b51b01c10f09f2fbd72f3a7d75272ce950a277e7d5d --conditioner-sha256 0e66c64baa9c4bb1e1299aeac2a4c4668204674677931398640caa4baaffd5d1 --trusted-source --words-from-pack data/openfs-packed --word-set train --train-words 50 --trials 3 --seed 20261019 --output data/openfs-synthetic-pilot --report data/openfs-synthetic-pilot.report.json

Real runs (word choice is documented in the provenance file and in
docs/synthetic-data-2026-10-19.md; never a word the validation or test transcripts contain, except
the 26 single letters):

  ... --word-set train jz --quality-rule prefix --train-words 24000 --jz-words 2400 --trials 3 --output data/openfs-synthetic-words --report data/openfs-synthetic-words.report.json
  ... --word-set letters --quality-rule prefix --trials 12 --output data/openfs-synthetic-letters --report data/openfs-synthetic-letters.report.json

Nothing here runs at import time and nothing under model/ may import this script: the guard is
tests/test_openfs_synthetic.py::AppRuntimeFirewall.
"""
import argparse
from functools import partial
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import string
import sys
import time
import types

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.recognition_metrics import ErrorTally
from scripts.prepare_openfs_training import UPSTREAM_CHARS

LICENSE_QUOTE = ('Commercial use of the source code, model checkpoints, or derivative works is '
                 'strictly prohibited')
WORD_TOKEN = re.compile('[A-Z]+')
MOVEMENT_LETTERS = 'JZ'
MAX_WORD_LENGTH = 12  # the released recognizer decodes at most max_len=32 tokens; FSboard words are shorter
OUTPUT_PREFIX = 'openfs-synthetic'  # this script may only ever write into data/openfs-synthetic*/


class UpstreamDependencyMissing(RuntimeError):
    """Raised when upstream code reaches for a module this environment stubbed out."""


def sha256_of(path, buffer_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(partial(handle.read, buffer_size), b''):
            digest.update(block)
    return digest.hexdigest()


def upstream_dependency_stubs():
    """Install loud stubs for upstream module-level imports this venv lacks; returns what was faked."""
    installed = {}

    def missing(name):
        if name in sys.modules:
            return False
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):     # find_spec trips over injected stub modules without __spec__
            return False
    if missing('tqdm'):
        module = types.ModuleType('tqdm')
        module.__spec__ = importlib.machinery.ModuleSpec('tqdm', loader=None)
        module.tqdm = lambda iterable=(), *args, **kwargs: iter(iterable)
        sys.modules['tqdm'] = module
        installed['tqdm'] = 'tqdm.tqdm is an iterator passthrough: a progress bar, no numeric effect'
    if missing('scipy'):
        scipy = types.ModuleType('scipy')
        scipy.__spec__ = importlib.machinery.ModuleSpec('scipy', loader=None)
        ndimage = types.ModuleType('scipy.ndimage')
        ndimage.__spec__ = importlib.machinery.ModuleSpec('scipy.ndimage', loader=None)

        def blocked_zoom(*args, **kwargs):
            raise UpstreamDependencyMissing('scipy.ndimage.zoom is upstream pose resampling '
                                            '(training augmentation); the generation path must not call it')
        ndimage.zoom = blocked_zoom
        scipy.ndimage = ndimage
        sys.modules['scipy'] = scipy
        sys.modules['scipy.ndimage'] = ndimage
        installed['scipy'] = 'scipy.ndimage.zoom raises: resampling is training-time augmentation'
    if missing('sklearn'):
        sklearn = types.ModuleType('sklearn')
        sklearn.__spec__ = importlib.machinery.ModuleSpec('sklearn', loader=None)
        preprocessing = types.ModuleType('sklearn.preprocessing')
        preprocessing.__spec__ = importlib.machinery.ModuleSpec('sklearn.preprocessing', loader=None)

        def blocked(name):
            raise UpstreamDependencyMissing(f'sklearn.preprocessing.{name} belongs to the upstream '
                                            'CTC CSV helpers, not to generation')
        preprocessing.__getattr__ = blocked
        sklearn.preprocessing = preprocessing
        sys.modules['sklearn'] = sklearn
        sys.modules['sklearn.preprocessing'] = preprocessing
        installed['sklearn'] = 'sklearn.preprocessing attribute access raises: only the CSV CTC helpers use it'
    return installed


def import_upstream(source):
    """Import the released classes; only ever called after the trusted-source opt-in is checked."""
    source = Path(source).resolve()
    if not (source / 'lib').is_dir():
        raise ValueError(f'{source} is not an OpenFS source checkout (no lib/ directory).')
    upstream_dependency_stubs()
    sys.path.insert(0, str(source))
    from lib.datasets.fingerspelling import WordDataset
    from lib.models.diffusion import Diffusion
    from lib.models.generator_df import EncoderTransformer
    from lib.models.recognizer import EncoderDecoderTransformer
    from lib.utils.dataloader import paired_collate_word_dataset_fn
    from lib.utils.dataset import normalize_torch
    from lib.utils.seed import set_all_random_seed
    from lib.utils.size import count_parameters
    from lib.utils.text_ctc_utils import get_autoreg_vocab, invert_to_chars
    return dict(WordDataset=WordDataset, Diffusion=Diffusion, EncoderTransformer=EncoderTransformer,
                EncoderDecoderTransformer=EncoderDecoderTransformer,
                paired_collate_word_dataset_fn=paired_collate_word_dataset_fn,
                normalize_torch=normalize_torch, set_all_random_seed=set_all_random_seed,
                count_parameters=count_parameters, get_autoreg_vocab=get_autoreg_vocab,
                invert_to_chars=invert_to_chars)


def transcript_tokens(clips):
    """Uppercase letter runs a pack's transcripts contain, exactly as the pack stores them."""
    tokens = set()
    for row in clips:
        tokens.update(WORD_TOKEN.findall(str(row['text']).upper()))
    return tokens


def word_pool(tokens, *, min_length, max_length, require_movement, exclude):
    """Letters-only tokens from our own pack, filtered by length, movement letter and held-out text."""
    movement = set(MOVEMENT_LETTERS)
    return sorted(word.lower() for word in tokens
                  if word.isalpha() and min_length <= len(word) <= max_length
                  and word not in exclude
                  and bool(movement & set(word)) == bool(require_movement))


def select_words(pool, count, seed):
    """Deterministic prefix of one seeded permutation, so a smaller word count is a subset of a bigger one."""
    if count >= len(pool):
        return sorted(pool)
    order = np.random.default_rng(seed).permutation(len(pool))
    return sorted(pool[int(index)] for index in order[:int(count)])


def derive_word_lists(pack, sets, *, train_words, jz_words, seed, exclude_splits=('validation', 'test')):
    """Choose word sets from our own packed FSboard transcripts, with the held-out text excluded.

    letters is the one exception to the held-out filter: the app's known hole is J and Z, which are
    single movement letters, and the 26 single letters are also the atoms every word is spelled from.
    """
    pack = Path(pack)
    train_tokens = transcript_tokens(json.loads((pack / 'train_clips.json').read_text(encoding='utf-8')))
    held = set()
    for split in exclude_splits:
        path = pack / f'{split}_clips.json'
        if path.is_file():
            held |= transcript_tokens(json.loads(path.read_text(encoding='utf-8')))
    pools = dict(letters=[char.lower() for char in string.ascii_uppercase],
                 jz=word_pool(train_tokens, min_length=3, max_length=MAX_WORD_LENGTH,
                              require_movement=True, exclude=held),
                 train=word_pool(train_tokens, min_length=3, max_length=MAX_WORD_LENGTH,
                                 require_movement=False, exclude=held))
    limits = dict(letters=len(pools['letters']), jz=jz_words, train=train_words)
    chosen = {name: select_words(pools[name], limits[name], seed) for name in sets}
    counts = {name: dict(pool=len(pools[name]), chosen=len(chosen[name])) for name in sets}
    return chosen, counts, sorted(held)


def read_word_files(paths, exclude_paths, max_length):
    """Canonical word lists from files: lowercase, alphabet-checked, deduped, sorted, held-out text dropped."""
    wanted = set(UPSTREAM_CHARS)
    excluded = set()
    for path in exclude_paths:
        excluded.update(token.lower() for token in Path(path).read_text(encoding='utf-8').split())
    words = set()
    for path in paths:
        for token in Path(path).read_text(encoding='utf-8').split():
            word = token.lower()
            if not word or word in excluded or len(word) > max_length or set(word) - wanted:
                continue
            words.add(word)
    return sorted(words)


def provenance_for(directory, expected_sha256, label):
    """Digest the checkpoint, cross-check it against PROVENANCE.json when one sits next to it."""
    directory = Path(directory)
    artifact = directory / 'best_model.pth'
    if not artifact.is_file():
        raise ValueError(f'{label} checkpoint is missing: {artifact}')
    digest = sha256_of(artifact)
    if not expected_sha256:
        raise ValueError(f'{label} sha256 must be pinned to run: pass --{label.lower()}-sha256.')
    if digest != expected_sha256:
        raise ValueError(f'{label} sha256 {digest} does not match the pinned {expected_sha256}.')
    record = dict(path=str(directory.resolve()), artifact=artifact.name, artifact_sha256=digest,
                  artifact_bytes=artifact.stat().st_size)
    note = directory / 'PROVENANCE.json'
    if note.is_file():
        document = json.loads(note.read_text(encoding='utf-8'))
        recorded = document.get('artifact_sha256')
        if recorded and recorded != digest:
            raise ValueError(f'{label} checkpoint does not match its own PROVENANCE.json '
                             f'({recorded}); refusing to generate.')
        record['provenance_file'] = str(note)
        for key in ('upstream_commit', 'google_drive_id', 'license', 'requires_at_generation_time'):
            if key in document:
                record[key] = document[key]
    return record


def model_kwargs(config, section):
    return {key: value for key, value in config[section].items()
            if not key.startswith('_') and key != 'ckpt_path'}


def transcribe(upstream, output_ids, index, inverse_vocabulary):
    """Released token-to-text decode of one generated clip; a failed decode is a rejection, not a crash."""
    try:
        return ''.join(upstream['invert_to_chars'](output_ids[index:index + 1, 1:-1].cpu(), inverse_vocabulary))
    except Exception as error:
        return f'\ufffddecode-failed:{type(error).__name__}'


def accepted_by(rule, truth, predicted):
    """Two keep/drop rules, both model-derived.

    exact keeps a clip only when the whole transcription is the word. The released decoder is greedy
    with max_len=32 and frequently never emits EOS on generated poses, so it appends hallucinated
    characters after a correctly read word; on single letters exact then keeps almost nothing, which
    would hollow out exactly the J/Z experiment. prefix keeps a clip when the transcription starts
    with the word, accepting that tail. Which rule ran is recorded, and both rates are always
    reported, so a prefix-kept pack is never mistaken for an exactly-transcribed one.
    """
    if rule == 'exact':
        return predicted == truth
    if rule == 'prefix':
        return predicted.startswith(truth) and bool(truth)
    raise ValueError(f'unknown quality rule {rule!r}.')


def quality_block(tallies, groups, matches, rule, clips):
    """Per-set and overall recognition-on-generated numbers, plus both keep-rule rates."""
    accepted = sum(1 for clip in clips if clip['accepted'])
    exact_total = sum(1 for clip in clips if clip['exact_match'])
    prefix_total = sum(1 for clip in clips if clip['prefix_match'])
    return dict(overall=tallies['overall'].summary(),
                groups={name: tally.summary() for name, tally in groups.items()},
                rule=rule, accepted_clips=accepted, rejected_clips=len(clips) - accepted,
                pass_rate=accepted / len(clips) if clips else None,
                exact_match_clips=exact_total, prefix_match_clips=prefix_total,
                exact_match_rate=exact_total / len(clips) if clips else None,
                prefix_match_rate=prefix_total / len(clips) if clips else None,
                match_rate_by_group={name: dict(exact=values[0] / len(clips) if clips else None,
                                                prefix=values[1] / len(clips) if clips else None)
                                     for name, values in matches.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--source', type=Path, help='official OpenFS source checkout (upstream imports need --trusted-source)')
    parser.add_argument('--trusted-source', action='store_true',
                        help='explicitly acknowledge loading upstream code and research checkpoints')
    parser.add_argument('--generator', type=Path, default=ROOT / 'data/openfs-checkpoints/generator_df_fsboard')
    parser.add_argument('--conditioner', type=Path, default=ROOT / 'data/openfs-checkpoints/recognizer_fsboard',
                        help='released recognizer named by the generator config: it transcribes clips, it does not condition them')
    parser.add_argument('--generator-sha256', help='pin the released generator digest; the run refuses without it')
    parser.add_argument('--conditioner-sha256', help='pin the released conditioner digest; the run refuses without it')
    parser.add_argument('--words-from-pack', type=Path, help='derive word sets from a packed FSboard dataset')
    parser.add_argument('--word-set', dest='word_set', nargs='+', choices=['train', 'jz', 'letters'], default=['train'])
    parser.add_argument('--train-words', type=int, default=0)
    parser.add_argument('--jz-words', type=int, default=0)
    parser.add_argument('--words', type=Path, action='append', default=[], help='word list file (repeatable)')
    parser.add_argument('--exclude-words', type=Path, action='append', default=[])
    parser.add_argument('--max-word-length', type=int, default=MAX_WORD_LENGTH)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--quality-rule', choices=['exact', 'prefix'], default='exact',
                        help='exact: the transcription is the whole word; prefix: it starts with the word. '
                             'The greedy released decoder often skips EOS on generated poses, so exact '
                             'keeps very little on single letters; both rates are always reported')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit-clips', type=int, help='cap generated clips (pilots); whole words only')
    parser.add_argument('--min-frames', type=int, default=3, help='frames per letter, as scripts/generate/generate_data_df.sh')
    parser.add_argument('--max-frames', type=int, default=10)
    parser.add_argument('--min-transition-frames', type=int, default=2)
    parser.add_argument('--max-transition-frames', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    if not args.trusted_source:
        parser.error('Loading upstream OpenFS code and research checkpoints needs --trusted-source.')
    if not args.source:
        parser.error('--source (official OpenFS checkout) is required together with --trusted-source.')
    if not args.output.name.startswith(OUTPUT_PREFIX):
        raise ValueError(f'--output must sit in a directory named {OUTPUT_PREFIX}*; got {args.output.name}.')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'{args.output} already has content; generations are never written over.')
    for label, directory, digest in (('Generator', args.generator, args.generator_sha256),
                                     ('Conditioner', args.conditioner, args.conditioner_sha256)):
        if not digest:
            parser.error(f'--{label.lower()}-sha256 is required so every pack carries a pinned provenance.')
    if not (args.words or args.words_from_pack):
        parser.error('Pass --words FILE or --words-from-pack PACK.')

    generator_provenance = provenance_for(args.generator, args.generator_sha256, 'Generator')
    conditioner_provenance = provenance_for(args.conditioner, args.conditioner_sha256, 'Conditioner')
    config_path = args.generator / '.hydra/config.yaml'
    if not config_path.is_file():
        raise ValueError(f'{config_path} is missing; the released generator config is part of the checkpoint.')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    characters = config['decode']['chars']

    word_lists, word_counts, held_tokens = ({}, {}, [])
    if args.words_from_pack:
        word_lists, word_counts, held_tokens = derive_word_lists(args.words_from_pack, args.word_set,
                                                                 train_words=args.train_words,
                                                                 jz_words=args.jz_words, seed=args.seed)
    else:
        word_lists = dict(provided=read_word_files(args.words, args.exclude_words, args.max_word_length))
        word_counts = dict(provided=dict(pool=len(word_lists['provided']), chosen=len(word_lists['provided'])))
    set_of_word = {}
    for name in ('letters', 'jz', 'train', 'provided'):
        for word in word_lists.get(name, []):
            set_of_word.setdefault(word, name)
    words = sorted(set_of_word)
    if not words:
        raise ValueError('No words survived the filters; nothing to generate.')

    torch.set_num_threads(4)
    stubs = upstream_dependency_stubs()
    upstream = import_upstream(args.source)
    source_commit = None
    try:
        import subprocess
        source_commit = subprocess.check_output(['git', '-C', str(args.source.resolve()), 'rev-parse', 'HEAD'],
                                                text=True).strip()
    except Exception:
        source_commit = generator_provenance.get('upstream_commit')

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'words').mkdir(parents=True, exist_ok=True)
    written_lists = {}
    for name, chosen in word_lists.items():
        path = args.output / 'words' / f'{name}.txt'
        path.write_text(''.join(f'{word}\n' for word in chosen), encoding='utf-8')
        written_lists[name] = dict(path=str(path), words=len(chosen), sha256=sha256_of(path))
    (args.output / 'words' / 'generated-union.txt').write_text(''.join(f'{word}\n' for word in words), encoding='utf-8')

    vocabulary, inverse_vocabulary, char_list = upstream['get_autoreg_vocab'](characters)
    upstream['set_all_random_seed'](args.seed)  # same seeding path the released generator script uses
    dataset = upstream['WordDataset'](vocab=vocabulary, text_paths=[str(args.output / 'words' / 'generated-union.txt')],
                                      min_frames_num=args.min_frames, max_frames_num=args.max_frames,
                                      min_transition_frames_num=args.min_transition_frames,
                                      max_transition_frames_num=args.max_transition_frames,
                                      use_frame_label_clean=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=upstream['paired_collate_word_dataset_fn'])
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu')
    diffusion = upstream['Diffusion'](**model_kwargs(config, 'diffusion')).to(device)
    generator = upstream['EncoderTransformer'](char_size=len(char_list) + 1,
                                               **model_kwargs(config, 'generator')).to(device)
    generator.load_state_dict(torch.load(args.generator / 'best_model.pth', map_location=device, weights_only=True))
    generator.eval()
    recognizer = upstream['EncoderDecoderTransformer'](char_size=len(char_list) + 1,
                                                       **model_kwargs(config, 'recognizer')).to(device)
    recognizer.load_state_dict(torch.load(args.conditioner / 'best_model.pth', map_location=device, weights_only=True))
    recognizer.eval()

    tallies = dict(overall=ErrorTally())
    matches = dict(overall=[0, 0])
    groups = {}
    clips, pose_arrays, label_arrays = [], [], []
    frames = offset = seen = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for trial in range(args.trials):
            for batch in loader:
                if args.limit_clips and seen + len(batch['word']) > args.limit_clips:
                    break
                frame_idx = batch['frame_idx_gen'].to(device)
                frame_label = batch['frame_label_gen'].to(device)
                pseudo = frame_label.clone()
                pseudo[pseudo == -100] = len(char_list) + 1
                poses = diffusion.sampling(generator, pseudo, frame_idx)
                for index in range(len(batch['word'])):
                    keep = frame_idx[index] != 1024
                    poses[index:index + 1, keep, :, :2] = upstream['normalize_torch'](
                        poses[index:index + 1, keep, :, :2], normalize_value=0.5)
                normalized = upstream['normalize_torch'](poses[..., :2], normalize_value=0.5)
                transcription_ids = recognizer.generate(normalized, torch.zeros_like(frame_idx), frame_idx,
                                                        bos_token_id=len(char_list), eos_token_id=0)['output_ids']
                for index, word in enumerate(batch['word']):
                    keep = frame_idx[index] != 1024
                    pose = poses[index][keep].cpu().numpy().astype(np.float32)
                    labels = frame_label[index][keep].cpu().numpy().astype(np.int16)
                    guess = transcribe(upstream, transcription_ids, index, inverse_vocabulary)
                    truth = str(word).upper()
                    predicted = guess.upper()
                    exact_match = predicted == truth
                    prefix_match = bool(truth) and predicted.startswith(truth)
                    accepted = accepted_by(args.quality_rule, truth, predicted)
                    name = f'trial{trial}-{word}'
                    tally = groups.setdefault(set_of_word.get(word, 'provided'), ErrorTally())
                    tallies['overall'].add(truth, predicted)
                    tally.add(truth, predicted)
                    matches['overall'][0] += int(exact_match)
                    matches['overall'][1] += int(prefix_match)
                    group_matches = matches.setdefault(set_of_word.get(word, 'provided'), [0, 0])
                    group_matches[0] += int(exact_match)
                    group_matches[1] += int(prefix_match)
                    clips.append(dict(name=name, word=truth, set=set_of_word.get(word, 'provided'), trial=trial,
                                      start=offset, end=offset + len(pose), transcription=predicted,
                                      accepted=bool(accepted), exact_match=exact_match, prefix_match=prefix_match,
                                      quality_filter=f'released recognizer transcription, {args.quality_rule} '
                                                   f'word match required'))
                    pose_arrays.append(pose)
                    label_arrays.append(labels)
                    offset += len(pose)
                    frames += len(pose)
                    seen += 1
                if args.limit_clips and seen >= args.limit_clips:
                    break
            if args.limit_clips and seen >= args.limit_clips:
                break
    elapsed = time.perf_counter() - started

    poses_path = args.output / 'poses.npy'
    flat = np.lib.format.open_memmap(poses_path, mode='w+', dtype=np.float32, shape=(frames, 21, 3))
    labels_flat = np.lib.format.open_memmap(args.output / 'frame_labels.npy', mode='w+', dtype=np.int16,
                                            shape=(frames,))
    position = 0
    for clip, pose, labels in zip(clips, pose_arrays, label_arrays):
        flat[position:position + len(pose)] = pose
        labels_flat[position:position + len(pose)] = labels
        position += len(pose)
    flat.flush()
    labels_flat.flush()
    del flat, labels_flat, pose_arrays, label_arrays
    (args.output / 'clips.json').write_text(json.dumps(clips), encoding='utf-8')

    accepted = sum(1 for clip in clips if clip['accepted'])
    document = dict(kind='openfs-synthetic-generation', research_only=True,
                    license=dict(quote=LICENSE_QUOTE, source='data/external/openfs/source/LICENSE',
                                 note='generated landmark sequences are a derivative work of a research '
                                      'checkpoint and must never train anything shipped in the app'),
                    generator=generator_provenance, conditioner=conditioner_provenance,
                    upstream_source=str(Path(args.source).resolve()), upstream_commit=source_commit,
                    sampler=dict(class_name=f'{upstream["Diffusion"].__module__}.{upstream["Diffusion"].__name__}',
                                 sampling_method=f'{type(diffusion).sampling.__module__}.'
                                                 f'{type(diffusion).sampling.__qualname__}',
                                 dataset_class_call='diffusion.sampling(generator, frame_label, frame_idx)',
                                 **model_kwargs(config, 'diffusion'),
                                 generator_architecture=model_kwargs(config, 'generator'),
                                 conditioner_architecture=model_kwargs(config, 'recognizer'),
                                 generator_parameters=upstream['count_parameters'](generator),
                                 conditioner_parameters=upstream['count_parameters'](recognizer)),
                    dataset=dict(class_name=f'{upstream["WordDataset"].__module__}.{upstream["WordDataset"].__name__}',
                                 collate_fn='lib.utils.dataloader.paired_collate_word_dataset_fn',
                                 use_frame_label_clean=True, alphabet=characters,
                                 min_frames_num=args.min_frames, max_frames_num=args.max_frames,
                                 min_transition_frames_num=args.min_transition_frames,
                                 max_transition_frames_num=args.max_transition_frames),
                    seed=args.seed, trials=args.trials, batch_size=args.batch_size, device=str(device),
                    word_lists=written_lists, word_counts=word_counts, words_generated=len(words),
                    held_out_tokens_excluded=len(held_tokens),
                    dependency_stubs=stubs,
                    clips=len(clips), frames=frames,
                    timing=dict(seconds=round(elapsed, 2), clips_per_second=round(len(clips) / elapsed, 3),
                                frames_per_second=round(frames / elapsed, 3),
                                peak_cuda_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20)
                                if device.type == 'cuda' else None),
                    quality_filter=dict(kind='model-derived quality filter (released recognizer transcription), '
                                              'not a human check',
                                        description=f'keep a clip when the released recognizer transcribes it with '
                                                    f'a {args.quality_rule} word match',
                                        **quality_block(tallies, groups, matches, args.quality_rule, clips)))
    (args.output / 'provenance.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    report = args.report or args.output.with_suffix('.report.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(json.dumps(dict(clips=len(clips), frames=frames, accepted=accepted,
                          exact_matches=document['quality_filter']['exact_match_clips'],
                          prefix_matches=document['quality_filter']['prefix_match_clips'],
                          cer=document['quality_filter']['overall']['cer'],
                          **{key: document['timing'][key] for key in ('seconds', 'clips_per_second', 'peak_cuda_mib')},
                          report=str(report))), flush=True)


if __name__ == '__main__':
    main()
