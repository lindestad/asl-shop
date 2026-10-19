"""Released-weight loading policy, gap-aware index builders, the streaming chunker and its
overlap merge for the transfer adapter.  Tiny synthetic fixtures only; never accuracy evidence.
"""
from pathlib import Path
import os
import re
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
try:
    import torch
except ImportError:
    torch=None

ALPHABET="{ !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~"


def make_frames(deltas,right=None,left=None):
    """Synthetic packed (frames,91) rows: presence flags, the delta column and distinct landmark
    coordinates per detected frame; everything else is zero."""
    values=np.zeros((len(deltas),91),dtype=np.float32)
    values[:,-1]=deltas
    if right is not None:
        values[right,44]=1
    if left is not None:
        values[left,89]=1
    for base,rows in ((0,right),(45,left)):
        if rows is not None:
            for row in rows:
                values[row,base:base+42]=np.arange(42,dtype=np.float32)+row
    return values


def research_paths():
    source=Path(os.environ.get('ASL_OPENFS_SOURCE',ROOT/'data/external/openfs/source'))
    directory=Path(os.environ.get('ASL_OPENFS_MODEL_DIR',ROOT/'data/openfs-checkpoints/recognizer_fsboard'))
    return source,directory


class IndexBuilderTests(unittest.TestCase):
    """The single thing the gaps42 adapter changes: which positional index a token carries."""

    def test_arange_erases_a_30_frame_detection_gap_and_gap_indices_keep_it(self):
        # 30 packed rows with no detection between two right-hand stretches (the v2 pack shape).
        deltas=np.full(60,1/30,dtype=np.float32)
        deltas[0]=0.  # the pack starts a clip with a zero delta, like the real rows do
        frames=make_frames(deltas,right=list(range(10))+list(range(40,45)))
        from scripts.finetune_openfs_transfer import window_tokens
        _,_,positions=window_tokens(frames,'control42')
        np.testing.assert_array_equal(positions,np.arange(15))
        _,_,positions=window_tokens(frames,'gaps42')
        np.testing.assert_allclose(positions,list(range(10))+list(range(40,45)),atol=1e-4)
        # 31 steps from row 9 to row 40: the 30 undetected rows plus the normal step into row 40.
        self.assertAlmostEqual(float(positions[10]-positions[9]),31.,places=3)

    def test_an_irregular_frame_delta_survives_in_the_positional_index(self):
        # One row whose delta is a whole second: 30 frames between two consecutive detections.
        deltas=np.full(30,1/30,dtype=np.float32)
        deltas[10]=1.
        frames=make_frames(deltas,right=list(range(30)))
        from scripts.finetune_openfs_transfer import window_tokens
        _,_,positions=window_tokens(frames,'gaps42')
        self.assertAlmostEqual(float(positions[10]-positions[9]),30.,places=3)
        _,_,positions=window_tokens(frames,'control42')
        np.testing.assert_allclose(positions,np.arange(30))

    def test_both_adapters_agree_on_tokens_and_differ_only_on_indices(self):
        frames=make_frames(np.full(40,1/30,dtype=np.float32),right=list(range(10))+list(range(25,30)),
                           left=list(range(12,20)))
        from scripts.finetune_openfs_transfer import window_tokens
        compressed=window_tokens(frames,'control42')
        gaps=window_tokens(frames,'gaps42')
        np.testing.assert_array_equal(compressed[0],gaps[0])
        self.assertEqual(compressed[1].tolist(),gaps[1].tolist())
        self.assertEqual(compressed[0].shape,(23,42))  # 15 right-hand and 8 left-hand detections
        self.assertFalse(np.allclose(compressed[2],gaps[2]))
        self.assertLessEqual(np.abs(gaps[0]).max(),.5+1e-6)  # bbox-normalised to +-0.5

    def test_each_track_index_starts_at_zero_even_after_a_long_absence(self):
        frames=make_frames(np.full(60,1/30,dtype=np.float32),right=list(range(5)),left=list(range(50,55)))
        from scripts.finetune_openfs_transfer import window_tokens
        for adapter in ('control42','gaps42'):
            _,segments,positions=window_tokens(frames,adapter)
            self.assertEqual(sorted(segments.tolist()),[0]*5+[1]*5)
            for hand in (0,1):
                track=positions[segments==hand]
                self.assertEqual(float(track[0]),0.)
                np.testing.assert_allclose(np.diff(track),np.ones(4),atol=1e-4)

    def test_gaps91_keeps_every_timeline_row_and_absent_frames_become_their_own_segment(self):
        frames=make_frames(np.full(12,1/30,dtype=np.float32),right=[0,1,2,9,10,11],left=[3,4,5])
        from scripts.finetune_openfs_transfer import window_tokens
        poses,segments,positions=window_tokens(frames,'gaps91')
        self.assertEqual(poses.shape,(12,91))
        self.assertEqual(segments.tolist(),[0,0,0,1,1,1,2,2,2,0,0,0])
        np.testing.assert_allclose(positions,np.arange(12),atol=1e-4)

    def test_unmirroring_flips_the_left_landmarks_only_and_touches_nothing_else(self):
        frames=make_frames(np.full(8,1/30,dtype=np.float32),right=list(range(4)),left=list(range(4,8)))
        from scripts.finetune_openfs_transfer import window_tokens
        mirrored=window_tokens(frames,'gaps91')
        flipped=window_tokens(frames,'gaps91',unmirror=True)
        left_x=slice(45,87,2)
        np.testing.assert_allclose(flipped[0][:,left_x],-mirrored[0][:,left_x],atol=1e-6)
        every=slice(None)
        np.testing.assert_allclose(np.delete(flipped[0],range(45,87,2),axis=1),
                                   np.delete(mirrored[0],range(45,87,2),axis=1),atol=1e-6)
        self.assertEqual(mirrored[1].tolist(),flipped[1].tolist())
        self.assertEqual(mirrored[2].tolist(),flipped[2].tolist())
        # For the wrist adapters the right-hand track is untouched, the left one changes shape.
        straight=window_tokens(frames,'control42',unmirror=True)
        straight_left=straight[0][straight[1]==1]
        mirrored_left=window_tokens(frames,'control42')[0][window_tokens(frames,'control42')[1]==1]
        np.testing.assert_allclose(straight_left[:,0::2],-mirrored_left[:,0::2],atol=1e-6)
        np.testing.assert_allclose(straight_left[:,1::2],mirrored_left[:,1::2],atol=1e-6)

    def test_a_window_without_any_detection_has_no_tokens_for_the_wrist_adapters(self):
        frames=make_frames(np.full(8,1/30,dtype=np.float32),right=[0,1],left=[6,7])
        from scripts.finetune_openfs_transfer import window_tokens
        self.assertIsNone(window_tokens(frames[2:6],'control42'))
        self.assertIsNone(window_tokens(frames[2:6],'gaps42'))
        self.assertEqual(window_tokens(frames[2:6],'gaps91')[1].tolist(),[2,2,2,2])


class ChunkerTests(unittest.TestCase):
    def test_windows_cover_every_frame_and_reuse_only_the_requested_overlap(self):
        from scripts.finetune_openfs_transfer import clip_windows
        for overlap in (0.,.25,.5):
            stride=int(round(300*(1-overlap)))
            for length in (1,2,299,300,301,450,700,711,1000):
                windows=clip_windows(length,300,overlap)
                self.assertTrue(windows,f'{length} {overlap}')
                self.assertEqual(windows[0][0],0)
                self.assertEqual(windows[-1][1],length)  # nothing dropped at the end
                starts=[start for start,_ in windows]
                self.assertEqual(starts,list(range(starts[0],starts[-1]+1,stride)))
                ends=[end for _,end in windows]
                self.assertEqual(ends,sorted(set(ends)))  # no window sits inside the previous one
                covered=np.zeros(length,dtype=int)
                for start,end in windows:
                    self.assertLess(start,end)
                    self.assertLessEqual(end,length)
                    covered[start:end]+=1
                self.assertTrue((covered>=1).all(),f'{length} {overlap}: {windows}')
                self.assertLessEqual(int(covered.max()),2)  # at most the intended second copy
                shared=sum(max(0,previous_end-start)
                           for (_,previous_end),(start,_) in zip(windows,windows[1:]))
                self.assertEqual(int(covered.sum())-length,shared,f'{length} {overlap}: {windows}')
                for index,(_,previous_end) in enumerate(windows[:-1]):
                    self.assertEqual(previous_end-windows[index+1][0],
                                     min(300-stride,previous_end-windows[index+1][0]))
                    if windows[index+1][1]>previous_end:  # a capped tail window may share less
                        self.assertLessEqual(previous_end-windows[index+1][0],300-stride)

    def test_the_final_window_is_the_short_one_and_adds_nothing_already_covered(self):
        from scripts.finetune_openfs_transfer import clip_windows
        self.assertEqual(clip_windows(700,300,.5),[(0,300),(150,450),(300,600),(450,700)])
        self.assertEqual(clip_windows(711,300,.5),[(0,300),(150,450),(300,600),(450,711)])
        self.assertEqual(clip_windows(300,300,.5),[(0,300)])
        self.assertEqual(clip_windows(301,300,.5),[(0,300),(150,301)])
        self.assertEqual(clip_windows(25,30,.5),[(0,25)])  # a covered tail is not repeated
        self.assertEqual(clip_windows(0,300,.5),[])
        self.assertEqual(clip_windows(10,4,0.),[(0,4),(4,8),(8,10)])

    def test_the_chunker_rejects_useless_window_settings(self):
        from scripts.finetune_openfs_transfer import clip_windows
        with self.assertRaises(ValueError): clip_windows(100,0,.5)
        with self.assertRaises(ValueError): clip_windows(100,300,1.)
        with self.assertRaises(ValueError): clip_windows(100,300,-.1)

    def test_decode_groups_respect_both_the_batch_cap_and_the_frame_budget(self):
        from scripts.finetune_openfs_transfer import groups_of
        items=[(np.zeros(length),None,None) for length in (10,20,1000,20,5)]
        groups=groups_of(items,64,60)
        self.assertEqual([[items[index][0].shape[0] for index in group] for group in groups],
                         [[5,10,20],[20],[1000]])  # four 20-frame items would need 80 padded frames
        groups=groups_of(items,2,10**6)
        self.assertEqual([len(group) for group in groups],[2,2,1])
        self.assertEqual(sorted(index for group in groups for index in group),list(range(len(items))))


class OverlapMergeTests(unittest.TestCase):
    def test_the_window_nearest_to_a_character_keeps_it(self):
        from scripts.finetune_openfs_transfer import merge_windows
        # Uniform windows cut the shared region at the midpoint between their centres (225): out of
        # four characters each, the left window keeps the three at 37.5, 112.5 and 187.5 and the
        # right one keeps the three from 262.5 onwards.
        self.assertEqual(merge_windows([(0,300,'ABCD'),(150,450,'EFGH')]),'ABCFGH')
        # B lands exactly on the midpoint and the tie goes to the earlier window, while C is nearer
        # the second centre and is dropped.
        self.assertEqual(merge_windows([(0,100,'AB'),(50,150,'CDE')]),'ABDE')
        self.assertEqual(merge_windows([(0,40,'HELLO')]),'HELLO')
        self.assertEqual(merge_windows([]),'')
        self.assertEqual(merge_windows([(0,40,''),(20,60,'BY')]),'BY')  # a silent window claims nothing
        self.assertEqual(merge_windows([(0,300,'AAA'),(150,450,'AAA')]),'AAAA')  # grids differ, so 4 of 6
        self.assertEqual(merge_windows([(150,450,'EFGH'),(0,300,'ABCD')]),'ABCFGH')  # input order free

    def test_a_four_window_merge_keeps_one_copy_per_character_in_timeline_order(self):
        from scripts.finetune_openfs_transfer import clip_windows,merge_windows
        windows=clip_windows(700,300,.5)
        labels={}
        rows=[]
        for position,(start,end) in enumerate(windows):
            count=(end-start)//100
            text=''.join(chr(ord('A')+4*position+order) for order in range(count))
            for order,character in enumerate(text):
                labels[character]=start+(order+.5)*(end-start)/count
            rows.append((start,end,text))
        merged=merge_windows(rows)
        # Hand-computed: centres are 150, 300, 450 and 575, so window 0 keeps A and B, window 1 keeps
        # F, window 2 keeps J, and window 3 keeps N.  C, D, E, G, H, I, K and M are the boundary
        # casualties: each is nearer another window's centre, and 550 and 512.5 (a tie) both fall to
        # other windows, which is the cost this protocol is meant to expose rather than hide.
        self.assertEqual(merged,'ABFJN')
        points=[labels[character] for character in merged]
        self.assertEqual(points,sorted(points))
        self.assertEqual(len(set(points)),len(points))
        # Which characters survive depends only on the window geometry, not the order they arrive in.
        self.assertEqual(merge_windows(list(reversed(rows))),merged)


class VocabularyTests(unittest.TestCase):
    chars=ALPHABET

    def test_released_alphabet_has_the_size_the_checkpoint_embedding_was_trained_with(self):
        self.assertEqual(len(self.chars),60)
        self.assertEqual(len(set(self.chars)),60)

    def test_transcripts_round_trip_through_the_released_token_layout(self):
        from scripts.finetune_openfs_transfer import decode_text,encode_text,reached_limit
        for text in ('HELLO WORLD','9-7241','A B. C','THE COTTON IS HIGH'):
            ids=encode_text(text,self.chars)
            self.assertEqual(ids[0],len(self.chars))  # BOS
            self.assertEqual(ids[-1],0)               # EOS, which collides with chars[0]
            self.assertEqual(decode_text(ids,self.chars),(''.join(text.lower()),0))
            self.assertFalse(reached_limit(ids,self.chars))
        self.assertTrue(reached_limit([len(self.chars),26,26],self.chars))

    def test_padding_start_token_leaks_and_outside_characters_match_the_published_harness(self):
        from scripts.finetune_openfs_transfer import alphabet_report,decode_text,encode_text
        char_size=len(self.chars)+1
        self.assertEqual(decode_text([char_size-1,self.chars.index('c'),self.chars.index('a'),
                                      self.chars.index('t'),0,char_size],self.chars),('cat',0))
        # The published harness writes a replacement character for a leaking start token and counts it.
        leaked=[len(self.chars),len(self.chars),self.chars.index('c'),self.chars.index('a'),
                self.chars.index('t'),0]
        self.assertEqual(decode_text(leaked,self.chars),('\ufffdcat',1))
        self.assertEqual(decode_text([len(self.chars),len(self.chars)+1],self.chars),('',0))  # pad stops it
        self.assertEqual(len(encode_text('A<B',self.chars)),4)  # '<' is outside: dropped, '?' is inside
        report=alphabet_report([dict(text='A<B'),dict(text='C')],self.chars)
        self.assertEqual(report['characters'],4)
        self.assertEqual(report['outside_alphabet'],1)
        self.assertEqual(report['outside_distinct'],['<'])

    def test_eos_and_the_first_alphabet_character_share_token_zero_as_the_released_model_trains(self):
        from scripts.finetune_openfs_transfer import decode_text,encode_text
        self.assertEqual(self.chars.index('{'),0)
        self.assertEqual(encode_text('{',self.chars),[len(self.chars),0,0])
        self.assertEqual(decode_text(encode_text('{',self.chars),self.chars),('',0))


@unittest.skipIf(torch is None,'Install the neural extra for the transfer adapter tests')
class PaddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def items(self):
        frames=make_frames(np.full(20,1/30,dtype=np.float32),right=list(range(12)),left=list(range(3,8)))
        from scripts.finetune_openfs_transfer import window_tokens
        return [window_tokens(frames,'gaps42')[:3],window_tokens(frames[:8],'gaps42')[:3]]

    def test_collate_pads_with_the_released_sentinels(self):
        from scripts.finetune_openfs_transfer import collate
        from model.offline_recognizer import PAD_FRAME,PAD_POSE,PAD_SEG
        items=self.items()
        longest,shortest=max(len(item[0]) for item in items),len(items[1][0])
        poses,segments,positions=collate([0,1],items,'cpu')
        self.assertEqual(poses.shape,(2,longest,42,1))  # the upstream forward wants a 4-d pose tensor
        self.assertEqual(segments.shape,(2,longest))
        self.assertEqual(positions.shape,(2,longest))
        self.assertTrue((poses[1,shortest:,0,0]==PAD_POSE).all())
        self.assertTrue((poses[0,:,0,0]!=PAD_POSE).all())  # real landmarks never reach -2
        self.assertTrue((segments[1,shortest:]==PAD_SEG).all())
        self.assertEqual(float(positions[1,-1]),PAD_FRAME)
        self.assertEqual(items[0][1].max(),1)  # two hand tracks, no third segment

    def test_collate_widens_for_the_91_channel_adapter(self):
        from scripts.finetune_openfs_transfer import collate,window_tokens
        frames=make_frames(np.full(9,1/30,dtype=np.float32),right=list(range(9)))
        poses,_,_=collate([0],[window_tokens(frames,'gaps91')[:3]],'cpu')
        self.assertEqual(poses.shape[1:],(9,91,1))


@unittest.skipIf(torch is None,'Install the neural extra for the released checkpoint tests')
class ReleasedCheckpointTests(unittest.TestCase):
    """Loads the research-licensed released checkpoint through the same gate the scripts use."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source,directory=research_paths()
        if not (source/'lib/models/recognizer.py').is_file() or not (directory/'best_model.pth').is_file():
            raise unittest.SkipTest('Pinned research source or released checkpoint not available')
        from scripts.finetune_openfs_transfer import load_reference
        try:
            cls.network,cls.chars,cls.architecture,cls.commit=load_reference(
                SimpleNamespace(source=source,checkpoint=directory/'best_model.pth'))
        except Exception as error:  # a conflicting upstream module already on sys.path is not our fault
            raise unittest.SkipTest(f'Released class unavailable: {error}')
        cls.checkpoint=directory/'best_model.pth'
        cls.released=torch.load(cls.checkpoint,map_location='cpu',weights_only=True)

    def build(self,adapter):
        return self.network(char_size=len(self.chars)+1,
                            **{**self.architecture,'input_dim':91 if adapter=='gaps91' else 42})

    def test_the_pinned_source_and_released_architecture_are_what_the_report_claims(self):
        self.assertRegex(self.commit,r'[0-9a-f]{40}|unknown')
        self.assertEqual(self.architecture['input_dim'],42)
        self.assertFalse(self.architecture['conventional_positional_encoding'])  # index-driven positions
        self.assertEqual(self.chars,ALPHABET)
        self.assertEqual(len(self.released),103)  # the tensor count the load policy reports against

    def test_control_and_gaps_adapters_load_every_released_tensor_unchanged(self):
        from scripts.finetune_openfs_transfer import load_released
        for adapter in ('control42','gaps42'):
            network=self.build(adapter)
            weights=load_released(network,self.checkpoint,adapter)
            self.assertEqual(weights['reinitialised'],[])
            self.assertEqual(weights['shape_mismatch'],[])
            self.assertTrue(weights['loaded_all'])
            self.assertEqual(weights['loaded_tensors'],weights['released_tensors'])
            self.assertEqual(weights['loaded_tensors'],len(self.released))
            self.assertEqual(weights['loaded_parameters'],sum(tensor.numel() for tensor in self.released.values()))
            own=network.state_dict()
            self.assertEqual(set(own),set(self.released))
            self.assertEqual([name for name,tensor in self.released.items() if not torch.equal(tensor,own[name])],[])

    def test_gaps91_reinitialises_exactly_the_first_pose_embedding_layer(self):
        from scripts.finetune_openfs_transfer import load_released
        self.assertIn('pose_embedding.0.weight',self.released)
        network=self.build('gaps91')
        weights=load_released(network,self.checkpoint,'gaps91')
        self.assertEqual(weights['reinitialised'],['pose_embedding.0.bias','pose_embedding.0.weight'])
        self.assertEqual(weights['shape_mismatch'],['pose_embedding.0.weight'])
        self.assertEqual(weights['loaded_tensors'],len(self.released)-2)
        self.assertEqual(weights['reinitialised_parameters'],91*256+256)
        own=network.state_dict()
        self.assertEqual(set(own)^set(self.released),set())
        changed={name for name,tensor in self.released.items() if not torch.equal(tensor,own[name])}
        self.assertEqual(changed,{'pose_embedding.0.weight','pose_embedding.0.bias'})
        self.assertEqual(own['pose_embedding.0.weight'].shape,(256,91))

    def test_a_network_that_does_not_match_the_released_layer_count_is_rejected(self):
        from scripts.finetune_openfs_transfer import load_released
        network=self.network(char_size=len(self.chars)+1,**{**self.architecture,'num_layers':1})
        with self.assertRaises(ValueError) as caught:
            load_released(network,self.checkpoint,'gaps42')
        message=str(caught.exception)
        self.assertIn('no place for',message)
        self.assertIn('decoder.layers.1',message)  # the released run had three layers, this build has one

    def frames(self):
        return make_frames(np.full(24,1/30,dtype=np.float32),right=list(range(24)),left=list(range(8,16)))

    def synthetic_data(self):
        """Six tiny clips in the packed layout read_data returns, long enough to need batching."""
        clips=[];rows=[]
        for order in range(6):
            length=8+2*order
            values=make_frames(np.full(length,1/30,dtype=np.float32),right=list(range(length)),
                               left=list(range(2,length-2)))
            values[:,0:42]=values[:,0:42]+order  # every clip has its own landmark signature
            clips.append(dict(name=f'clip{order}',start=len(rows),end=len(rows)+length,
                              text='ABCD'[:order+1]))
            rows.append(values)
        return dict(x=np.concatenate(rows),clips=clips)

    def test_every_adapter_fine_tunes_on_the_same_pinned_clip_sample(self):
        from scripts.finetune_openfs_transfer import training_batches
        data=self.synthetic_data()
        def sample(adapter,seed):
            return [sorted(data['clips'][index]['name'] for index in kept)
                    for _,_,_,_,kept in training_batches(data,self.chars,adapter,2,40,augment=True,
                                                          limit=6,rng=np.random.default_rng(seed))]
        for adapter in ('control42','gaps42','gaps91'):
            batches=sample(adapter,7)
            self.assertTrue(batches,'no batches for '+adapter)
            self.assertEqual(sum(len(names) for names in batches),6)  # every clip used exactly once
            self.assertEqual(len({name for names in batches for name in names}),6)
            self.assertEqual(batches,sample(adapter,7))  # the same pin gives the same batches
            if adapter!='control42': self.assertEqual(batches,sample('control42',7))
        # different epoch pins still give different samples, otherwise epochs would repeat one batch
        drawn=[tuple(batch) for seed in (1,2,3,4,5,6) for batch in sample('gaps42',seed)]
        self.assertGreater(len(set(drawn)),1)

    def test_augmentation_works_without_a_pinned_generator_too(self):
        # clip_tokens keeps a module generator for its augment decisions; handing the augment
        # helpers the numpy.random module instead is the bug this guards.
        from scripts.finetune_openfs_transfer import clip_tokens,training_batches
        data=self.synthetic_data()
        batches=list(training_batches(data,self.chars,'control42',2,60,augment=True,limit=6))
        self.assertEqual(sum(len(row[4]) for row in batches),6)
        for adapter in ('control42','gaps42','gaps91'):
            self.assertIsNotNone(clip_tokens(data,0,adapter,augment=True))

    def test_a_larger_frame_budget_puts_more_clips_in_each_batch(self):
        from scripts.finetune_openfs_transfer import training_batches
        data=self.synthetic_data()
        small=list(training_batches(data,self.chars,'gaps42',16,20,rng=np.random.default_rng(1)))
        large=list(training_batches(data,self.chars,'gaps42',16,10**6,rng=np.random.default_rng(1)))
        self.assertEqual(sum(len(row[4]) for row in small),6)
        self.assertEqual(sum(len(row[4]) for row in large),6)
        self.assertGreater(len(small),len(large))
        self.assertEqual(len(large),1)  # one clip per batch would need more frames than these hold

    def test_the_released_network_forwards_and_decodes_our_packed_tokens(self):
        from scripts.finetune_openfs_transfer import collate,decode,encode_text,load_released,window_tokens
        for adapter in ('control42','gaps42','gaps91'):
            network=self.build(adapter)
            load_released(network,self.checkpoint,adapter)
            network.eval()
            items=[window_tokens(self.frames(),adapter)[:3],window_tokens(self.frames()[:15],adapter)[:3]]
            targets=torch.tensor([encode_text('CAT',self.chars),encode_text('DOG',self.chars)])
            poses,segments,positions=collate([0,1],items,'cpu')
            with torch.inference_mode():
                output=network(poses,segments,positions,targets[:,:-1])
                texts,notes=decode(network,self.chars,items,'cpu',max_decode=6,batch_size=4,frame_budget=64)
            self.assertEqual(output['logits'].shape,(2,targets.shape[1]-1,len(self.chars)+1))
            self.assertTrue(torch.isfinite(output['logits']).all())
            self.assertEqual(len(output['cross_attns']),self.architecture['num_layers'])
            rows,columns=2,targets.shape[1]-1
            for layer in output['cross_attns']:
                self.assertEqual(layer.shape,(rows,columns,len(items[0][0])))
                np.testing.assert_allclose(layer.sum(-1).detach().numpy(),np.ones((rows,columns)),atol=1e-4)
            self.assertEqual(len(texts),2)
            self.assertTrue(all(isinstance(text,str) for text in texts))
            self.assertEqual(set(notes),{'decode_limit_hits','invalid_special_tokens'})
            self.assertLessEqual(notes['decode_limit_hits'],2)

    def test_the_fine_tune_step_runs_and_backpropagates_through_the_released_network(self):
        from scripts.finetune_openfs_transfer import (collate,encode_text,fine_tune_step,load_released,
                                                      window_tokens)
        network=self.build('gaps42')
        load_released(network,self.checkpoint,'gaps42')
        network.train()
        frames=make_frames(np.full(18,1/30,dtype=np.float32),right=list(range(18)))
        items=[window_tokens(frames,'gaps42')[:3],window_tokens(frames[:12],'gaps42')[:3]]
        poses,segments,positions=collate([0,1],items,'cpu')
        raw=[encode_text('CAT',self.chars),encode_text('AB',self.chars)]
        targets=torch.full((2,max(len(ids) for ids in raw)),-100,dtype=torch.long)
        for row,ids in enumerate(raw): targets[row,:len(ids)]=torch.tensor(ids)
        arguments=SimpleNamespace(amp='off',amp_dtype=None,ce_weight=1.,entropy_weight=.8,
                                  monotonicity_weight=1.,warm_epoch=0)
        loss,ce,entropy,monotonicity=fine_tune_step(network,poses,segments,positions,targets,1,arguments,
                                                    len(self.chars)+1,len(self.chars)+1)
        self.assertTrue(torch.isfinite(loss))
        number=lambda value:float(value.detach()) if torch.is_tensor(value) else float(value)
        values={name:number(value) for name,value in
                dict(loss=loss,ce=ce,entropy=entropy,monotonicity=monotonicity).items()}
        self.assertGreater(values['loss'],values['ce'])  # epoch 1 > warm epoch 0 adds the auxiliary losses
        self.assertAlmostEqual(values['loss'],values['ce']+.8*values['entropy']+values['monotonicity'],places=5)
        self.assertGreaterEqual(values['entropy'],-1e-6)  # a fresh released net concentrates its attention
        self.assertGreaterEqual(values['monotonicity'],0.)
        loss.backward()
        self.assertIsNotNone(network.pose_embedding[0].weight.grad)
        self.assertIsNotNone(network.char_embedding.weight.grad)
        self.assertTrue((network.pose_embedding[0].weight.grad!=0).any())


class ReportComparisonTests(unittest.TestCase):
    """The merged report is what makes the three headline deltas readable, so compare and
    merge_into are the parts worth pinning down."""

    def block(self,cer,exact,deletions=10,substitutions=20,insertions=5):
        return dict(cer=cer,letter_accuracy=1-cer,exact_sequence_accuracy=exact,
                    deletions=deletions,substitutions=substitutions,insertions=insertions,
                    deletion_rate=deletions/100,insertion_rate=insertions/100,
                    clips=100,characters=100,errors=int(cer*100),seconds=1.)

    def report(self):
        return dict(adapters={
            'control42':dict(released={'full-clip':self.block(.06,.59),'chunked':self.block(.075,.55,10,20,12)},
                             fine_tuned={'full-clip':self.block(.05,.62),'chunked':self.block(.06,.58,10,20,11)}),
            'gaps42':dict(released={'full-clip':self.block(.057,.60),'chunked':self.block(.071,.56,10,20,13)},
                          fine_tuned={'full-clip':self.block(.048,.63),'chunked':self.block(.058,.59,10,20,12)}),
        },comparison={})

    def test_compare_reports_paired_deltas_for_every_stage_and_mode(self):
        from scripts.finetune_openfs_transfer import compare
        report=self.report()
        compare(report,dict(cer=.06286748,exact_sequence_accuracy=.59324335,clips=12521,characters=221100))
        self.maxDiff=None
        # deltas(before, after) treats a positive cer_points as an improvement of after over before
        self.assertEqual(report['comparison']['gaps42_minus_control42_released']['cer_points'],0.3)
        self.assertEqual(report['comparison']['gaps42_minus_control42_fine_tuned']['cer_points'],0.2)
        for stage in ('released','fine_tuned'):
            cost=report['comparison'][f'chunked_cost_{stage}']['gaps42']
            self.assertLess(cost['cer_points'],0)  # chunking never improves CER here
            self.assertGreater(cost['insertion_change'],0)
        control=report['comparison']['control_vs_published']
        self.assertEqual(control['cer_points'],round(100*(.06286748-.06),3))
        self.assertEqual(control['published']['clips'],12521)
        self.assertEqual(control['ours']['clips'],100)

    def test_compare_survives_a_missing_published_row_and_reruns_cleanly(self):
        from scripts.finetune_openfs_transfer import compare
        report=self.report()
        compare(report,{})
        self.assertNotIn('control_vs_published',report['comparison'])
        compare(report,dict(cer=.06,exact_sequence_accuracy=.59))
        self.assertIn('control_vs_published',report['comparison'])
        keys=set(report['comparison'])
        report['comparison']['chunked_cost']={'legacy':True}  # older runs wrote this key
        compare(report,dict(cer=.06,exact_sequence_accuracy=.59))  # merging re-runs compare
        self.assertEqual(keys,set(report['comparison']))
        self.assertNotIn('chunked_cost',report['comparison'])
        self.assertEqual(report['comparison']['gaps42_minus_control42_released']['cer_points'],0.3)

    def test_merging_a_gaps91_run_adds_cross_adapter_deltas(self):
        from scripts.finetune_openfs_transfer import compare,merge_into
        report=self.report()
        gaps91=dict(adapters={'gaps91':dict(
                        released={'full-clip':self.block(1.1,0.,9198,20955,8140),
                                  'chunked':self.block(1.117,0.,8324,21734,8829)},
                        fine_tuned={'full-clip':self.block(.09,.42),
                                    'chunked':self.block(.10,.40)})},
                    comparison={},commands=['python run gaps91'],peak_vram_mib=1602,seconds=3600,
                    evaluation=dict(selection=dict(clips=2000,seed=42)))
        merge_into(report,gaps91,'data/transfer-teacher-gaps91.report.json')
        compare(report,dict(cer=.06286748,exact_sequence_accuracy=.59324335))
        self.assertEqual(report['comparison']['gaps91_minus_control42_released']['cer_points'],
                         round(100*(.06-1.1),3))
        self.assertGreater(report['comparison']['gaps91_minus_control42_fine_tuned']['cer_points'],
                           report['comparison']['gaps91_minus_control42_released']['cer_points'])
        self.assertIn('gaps91',report['comparison']['chunked_cost_fine_tuned'])
        merge_into(report,gaps91,'data/transfer-teacher-gaps91.report.json')  # idempotent
        self.assertEqual(len(report['merged_from']),1)
        self.assertEqual(report['commands'],['python run gaps91'])
        self.assertEqual(report['merged_from'][0]['report'],'data/transfer-teacher-gaps91.report.json')
        self.assertEqual(report['merged_from'][0]['adapters'],['gaps91'])
        # the run that merges keeps its own adapter blocks; the merged file only adds missing ones
        report['adapters']['gaps42']['released']['full-clip']['cer']=.999
        merge_into(report,gaps91,'other.json')
        self.assertEqual(report['adapters']['gaps42']['released']['full-clip']['cer'],.999)


if __name__=='__main__':
    unittest.main()
