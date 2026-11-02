"""Hand-computed loss values, dual-level encodings, index builders and augmentation
determinism for the offline recogniser.  Tiny fixtures only; never accuracy evidence.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
try:
    import torch
except ImportError:
    torch=None


def make_frames(deltas,right=None,left=None):
    """Synthetic packed (frames,91) rows: presence flags, the delta column and distinct
    landmark coordinates per detected frame; everything else is zero."""
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


class IndexBuilderTests(unittest.TestCase):
    def test_timeline_indices_keep_irregular_frame_gaps(self):
        deltas=np.array([0,1/30,1/30,4/30,1/30],dtype=np.float32)
        positions=timeline_indices_of(deltas)
        np.testing.assert_allclose(positions,[0,1,2,6,7],atol=1e-4)

    def test_gaps_variant_preserves_detected_frame_gap_compressed_does_not(self):
        # Right hand detected on rows 0 and 1, absent on 2-4, detected again on 5.
        deltas=np.array([0,1/30,1/30,1/30,1/30,1/30],dtype=np.float32)
        frames=make_frames(deltas,right=[0,1,5])
        from scripts.train_openfs_recognizer import compressed_tokens,timeline_tokens
        pose,segments,positions,text=timeline_tokens(frames,'CAT','gaps')
        np.testing.assert_allclose(positions,np.arange(6),atol=1e-4)  # the full timeline stays
        self.assertEqual(segments.tolist(),[0,0,2,2,2,0])  # the absent frames are their own segment
        pose,segments,positions,text=compressed_tokens(frames,'CAT')
        np.testing.assert_array_equal(positions,[0,1,2])  # detection order erases the gap
        self.assertEqual(segments.tolist(),[0,0,0])
        self.assertEqual(pose.shape,(3,42))

    def test_compressed_tokens_concatenate_both_hands_as_segments(self):
        deltas=np.array([0,1/30,1/30],dtype=np.float32)
        frames=make_frames(deltas,right=[0],left=[1,2])
        from scripts.train_openfs_recognizer import compressed_tokens
        pose,segments,positions,_=compressed_tokens(frames,'AB')
        self.assertEqual(segments.tolist(),[0,1,1])
        np.testing.assert_array_equal(positions,[0,0,1])  # arange restarts per hand
        self.assertLessEqual(np.abs(pose).max(),.5+1e-6)  # bbox-normalised to +-0.5


def timeline_indices_of(deltas):
    from scripts.train_openfs_recognizer import timeline_indices
    frames=np.zeros((len(deltas),91),dtype=np.float32)
    frames[:,-1]=deltas
    return timeline_indices(frames)


@unittest.skipIf(torch is None,'Install the neural extra for recogniser loss tests')
class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from model.offline_recognizer import attention_entropy_loss,soft_monotonicity_loss
        cls.entropy_loss=staticmethod(attention_entropy_loss)
        cls.monotonicity_loss=staticmethod(soft_monotonicity_loss)
        from model.neural import VOCAB
        cls.a=VOCAB.index('A')
        cls.b=VOCAB.index('B')
        from model.offline_recognizer import BOS,PAD_SEG
        cls.BOS=BOS
        cls.PAD_SEG=PAD_SEG

    def test_attention_entropy_matches_hand_computed_value(self):
        # One character, three tokens, two hand segments (0,0,1): attention [.25,.25,.5]
        # pools to masses [.5,.5]; each term is -.5 ln .5 = .3466 and their mean is also .3466.
        weights=torch.tensor([[[[.25,.25,.5]]]])  # (B,layers,chars,tokens)
        segments=torch.tensor([[0,0,1]])
        targets=torch.tensor([[self.BOS,self.a]])
        loss=self.entropy_loss(weights,segments,targets)
        self.assertAlmostEqual(float(loss),-0.5*np.log(.5),places=6)

    def test_attention_entropy_ignores_padded_characters_and_counts_zero_as_zero(self):
        weights=torch.tensor([[[[.25,.25,.5],[1/3,1/3,1/3]]]])  # second character is padding
        segments=torch.tensor([[0,0,1]])
        targets=torch.tensor([[self.BOS,self.a,-100]])
        loss=self.entropy_loss(weights,segments,targets)
        self.assertAlmostEqual(float(loss),-0.5*np.log(.5),places=6)
        sharp=torch.tensor([[[[0.,1.,0.]]]])  # all mass in one segment -> zero entropy
        loss=self.entropy_loss(sharp,torch.tensor([[1,0,1]]),torch.tensor([[self.BOS,self.a]]))
        self.assertAlmostEqual(float(loss),0.,places=6)

    def test_soft_monotonicity_penalises_backward_attention_only(self):
        segments=torch.tensor([[0,1,0]])
        targets=torch.tensor([[self.BOS,self.a,self.b]])
        forward=torch.tensor([[[[1.,0.,0.],[0.,0.,1.]]]])  # attention centroid advances
        self.assertAlmostEqual(float(self.monotonicity_loss(forward,segments,targets)),0.,places=6)
        backward=torch.tensor([[[[0.,0.,1.],[1.,0.,0.]]]])  # one character drifts back
        # ReLU mass differences [1,1,0] over 2 valid characters x 3 valid tokens.
        self.assertAlmostEqual(float(self.monotonicity_loss(backward,segments,targets)),2/6,places=6)

    def test_soft_monotonicity_masks_padded_tokens(self):
        segments=torch.tensor([[0,1,self.PAD_SEG]])
        targets=torch.tensor([[self.BOS,self.a,self.b]])
        backward=torch.tensor([[[[0.,0.,1.],[1.,0.,0.]]]])
        # Column 2 is padded, so only the two valid columns contribute over 2x2 valid cells.
        self.assertAlmostEqual(float(self.monotonicity_loss(backward,segments,targets)),2/4,places=6)


@unittest.skipIf(torch is None,'Install the neural extra for recogniser model tests')
class PositionalEncodingTests(unittest.TestCase):
    def test_same_index_gives_same_embedding_and_different_indices_differ(self):
        from model.offline_recognizer import positional_encoding_from_idx
        same=positional_encoding_from_idx(torch.tensor([3.0,3.0]),8)
        self.assertTrue(torch.equal(same[0],same[1]))
        apart=positional_encoding_from_idx(torch.tensor([3.0,4.0]),8)
        self.assertFalse(torch.allclose(apart[0],apart[1]))
        zero=positional_encoding_from_idx(torch.tensor([0.0]),8)
        self.assertTrue(torch.equal(zero[0,0::2],torch.zeros(4)))  # interleaved sin, cos
        self.assertTrue(torch.equal(zero[0,1::2],torch.ones(4)))
        one=positional_encoding_from_idx(torch.tensor([1.0]),8)
        self.assertAlmostEqual(float(one[0,0]),np.sin(1),places=6)
        self.assertAlmostEqual(float(one[0,1]),np.cos(1),places=6)


@unittest.skipIf(torch is None,'Install the neural extra (scripts import torch)')
class AugmentationTests(unittest.TestCase):
    def test_pose_augmentation_is_deterministic_under_a_seed(self):
        from scripts.train_openfs_recognizer import augment_pose
        pose=(np.random.default_rng(1).normal(size=(60,21,2))).astype(np.float32)
        first=augment_pose(pose.copy(),np.random.default_rng(7))
        second=augment_pose(pose.copy(),np.random.default_rng(7))
        np.testing.assert_array_equal(first,second)
        other=augment_pose(pose.copy(),np.random.default_rng(8))
        self.assertFalse(first.shape==other.shape and np.array_equal(first,other))
        # The seeded draws actually change the track (resample may also change its length).
        self.assertTrue(first.shape!=pose.shape or not np.array_equal(first,pose))

    def test_timeline_augmentation_is_deterministic_under_a_seed(self):
        from scripts.train_openfs_recognizer import augment_timeline
        frames=make_frames(np.full(40,1/30,dtype=np.float32),right=list(range(40)))
        first=augment_timeline(frames.copy(),np.random.default_rng(3))
        second=augment_timeline(frames.copy(),np.random.default_rng(3))
        np.testing.assert_array_equal(first,second)
        self.assertFalse(np.array_equal(first,augment_timeline(frames.copy(),np.random.default_rng(4))))


@unittest.skipIf(torch is None,'Install the neural extra for recogniser model tests')
class RecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self):
        from model.offline_recognizer import BOS,EOS,PAD_FRAME,PAD_POSE,PAD_SEG
        from model.neural import VOCAB
        torch.manual_seed(0)
        first=torch.cat((torch.randn(25,42),torch.full((5,42),float(PAD_POSE))))
        second=torch.cat((torch.randn(15,42),torch.full((15,42),float(PAD_POSE))))
        poses=torch.stack((first,second))
        segments=torch.stack((torch.tensor([0]*25+[PAD_SEG]*5),torch.tensor([1]*15+[PAD_SEG]*15)))
        positions=torch.stack((torch.cat((torch.arange(25,dtype=torch.float32),torch.full((5,),float(PAD_FRAME)))),
                               torch.cat((torch.arange(15,dtype=torch.float32),torch.full((15,),float(PAD_FRAME))))))
        full=[BOS,*[VOCAB.index(c) for c in 'HI'],EOS]
        short=[BOS,VOCAB.index('A'),EOS]
        targets=torch.full((2,5),-100,dtype=torch.long)
        targets[0,:len(full)]=torch.tensor(full)
        targets[1,:len(short)]=torch.tensor(short)
        return poses,segments,positions,targets

    def test_forward_shapes_attention_rows_and_gradients(self):
        from model.offline_recognizer import (Recognizer,attention_entropy_loss,soft_monotonicity_loss,
                                              PAD_CHAR)
        model=Recognizer(input_dim=42,d_model=32,heads=2,layers=1,feedforward=64,dropout=0.)
        poses,segments,positions,targets=self.batch()
        inputs=torch.where(targets[:,:-1]==-100,PAD_CHAR,targets[:,:-1])
        logits,weights=model(poses,segments,positions,inputs)
        self.assertEqual(logits.shape[:2],(2,4))
        self.assertEqual(logits.shape[-1],60)
        self.assertEqual(weights.shape,(2,1,4,30))
        self.assertTrue(torch.isfinite(weights).all())
        np.testing.assert_allclose(weights[0,0].sum(-1).detach().numpy(),np.ones(4),atol=1e-5)
        from torch.nn import functional as F
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets[:,1:].reshape(-1))
        loss=loss+attention_entropy_loss(weights,segments,targets)+soft_monotonicity_loss(weights,segments,targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.decoder[0].cross_attention.query.weight.grad)

    def test_generate_roundtrips_through_vocab(self):
        from model.offline_recognizer import Recognizer,decode_text,encode_text
        model=Recognizer(input_dim=42,d_model=32,heads=2,layers=1,feedforward=64,dropout=0.)
        model.eval()
        self.assertEqual(decode_text(encode_text('HI')[1:]),'HI')
        poses,segments,positions,_=self.batch()
        with torch.no_grad():
            decoded=model.generate(poses,segments,positions,max_chars=6)
        self.assertEqual(len(decoded),2)
        self.assertTrue(all(1<=len(row)<=6 for row in decoded))


if __name__=='__main__':
    unittest.main()
