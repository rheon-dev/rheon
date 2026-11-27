//! Counter-based Philox4x64-10 pseudorandom draw stream (Salmon et al. 2011).
//!
//! Philox4x64-10 is a keyed bijection over four 64-bit counter words under a
//! two-word 128-bit key, using 10 rounds of two 128-bit multiplications and a
//! fixed permutation with Weyl key schedule. It matches `numpy.random.Philox`
//! to the bit and requires no mutable generator state beyond the learned-row
//! count `n`.
//!
//! ## Stream layout
//!
//! ```text
//! key     = [seed, group_hash]            two u64
//! counter = [j, lane, n, slot]            four u64
//!   j     block index within one row's draws (4 u64 words per block);
//!         forests place the tree index in the high 32 bits: (tree << 32) | block
//!   lane  frozen domain tag:
//!           0  bandit policy uniforms
//!           1  bandit policy standard normals
//!           2  half-space tree construction (n = 0)
//!           3  random cut forest insertion cuts
//!   n     learned-row count before this row (only positive-weight learned rows
//!         advance n, so zero-weight rows and predict() leave state identical)
//!   slot  hash of the spec name (0 in standalone core use)
//! ```
//!
//! ## Cross-platform bit-exactness
//!
//! All word conversions use integer arithmetic, IEEE-754 operations (`+`, `-`,
//! `*`, `/`, `sqrt`), and `libm::log` (never the host C library's `log`), so
//! every uniform, index, and Wichura AS 241 standard normal is bit-identical
//! across macOS, Linux, and Windows on both arm64 and x86-64.

use serde::{Deserialize, Serialize};

const M0: u64 = 0xD2E7_470E_E14C_6C93;
const M1: u64 = 0xCA5A_8263_9512_1157;
const W0: u64 = 0x9E37_79B9_7F4A_7C15;
const W1: u64 = 0xBB67_AE85_84CA_A73B;
const ROUNDS: usize = 10;

/// 2^-53, exact power of two.
const TWO_M53: f64 = 1.0 / ((1u64 << 53) as f64);

/// Frozen Philox lane: bandit policy uniform draws (explore coin, arm choice).
pub const LANE_POLICY_UNIFORM: u64 = 0;
/// Frozen Philox lane: bandit policy standard normal draws (Thompson sampling).
pub const LANE_POLICY_NORMAL: u64 = 1;
/// Frozen Philox lane: half-space tree construction draws (`n = 0`).
pub const LANE_HST_BUILD: u64 = 2;
/// Frozen Philox lane: random cut forest per-insertion cut draws.
pub const LANE_RCF_CUT: u64 = 3;

/// One Philox4x64-10 block from a 256-bit counter and 128-bit key.
#[inline]
pub fn philox4x64(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = ctr;
    let [mut k0, mut k1] = key;
    for r in 0..ROUNDS {
        if r > 0 {
            k0 = k0.wrapping_add(W0);
            k1 = k1.wrapping_add(W1);
        }
        let p0 = u128::from(M0) * u128::from(c0);
        let p1 = u128::from(M1) * u128::from(c2);
        let hi0 = (p0 >> 64) as u64;
        let lo0 = p0 as u64;
        let hi1 = (p1 >> 64) as u64;
        let lo1 = p1 as u64;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
    }
    [c0, c1, c2, c3]
}

/// Uniform double in `[0, 1)`: the top 53 bits times `2^-53`, bit-identical
/// to `numpy.random.Generator.random`.
#[inline]
pub fn u01(w: u64) -> f64 {
    ((w >> 11) as f64) * TWO_M53
}

/// Uniform double strictly in `(0, 1)`: the top 52 bits with a low `1` bit
/// appended, times `2^-53` (an odd multiple of `2^-53`, exact in `f64`).
#[inline]
pub fn u01_open(w: u64) -> f64 {
    (((w >> 11) | 1) as f64) * TWO_M53
}

/// Uniform integer index in `0..k` via Lemire's multiply-shift
/// `((w as u128) * (k as u128)) >> 64` (bias at most `k / 2^64`).
///
/// Panics in debug builds if `k == 0`.
#[inline]
pub fn below(w: u64, k: usize) -> usize {
    debug_assert!(k > 0, "below requires k > 0");
    ((u128::from(w) * (k as u128)) >> 64) as usize
}

// Wichura (1988), Algorithm AS 241, PPND16 coefficients.
#[allow(clippy::excessive_precision)]
const AS241_A: [f64; 8] = [
    3.387_132_872_796_366_6e0,
    1.331_416_678_917_843_8e2,
    1.971_590_950_306_551_4e3,
    1.373_169_376_550_946_1e4,
    4.592_195_393_154_987e4,
    6.726_577_092_700_87e4,
    3.343_057_558_358_813e4,
    2.509_080_928_730_122_7e3,
];
const AS241_B: [f64; 8] = [
    1.0,
    4.231_333_070_160_091e1,
    6.871_870_074_920_579e2,
    5.394_196_021_424_751e3,
    2.121_379_430_158_659_6e4,
    3.930_789_580_009_271e4,
    2.872_908_573_572_194_3e4,
    5.226_495_278_852_855e3,
];
const AS241_C: [f64; 8] = [
    1.423_437_110_749_683_6e0,
    4.630_337_846_156_545e0,
    5.769_497_221_460_691e0,
    3.647_848_324_763_204_6e0,
    1.270_458_252_452_368_4e0,
    2.417_807_251_774_506e-1,
    2.272_384_498_926_918_5e-2,
    7.745_450_142_783_414e-4,
];
const AS241_D: [f64; 8] = [
    1.0,
    2.053_191_626_637_759e0,
    1.676_384_830_183_803_8e0,
    6.897_673_349_851e-1,
    1.481_039_764_274_800_8e-1,
    1.519_866_656_361_645_7e-2,
    5.475_938_084_995_345e-4,
    1.050_750_071_644_416_8e-9,
];
const AS241_E: [f64; 8] = [
    6.657_904_643_501_104e0,
    5.463_784_911_164_114e0,
    1.784_826_539_917_291_4e0,
    2.965_605_718_285_049e-1,
    2.653_218_952_657_612_3e-2,
    1.242_660_947_388_078_4e-3,
    2.711_555_568_743_487_6e-5,
    2.010_334_399_292_288e-7,
];
const AS241_F: [f64; 8] = [
    1.0,
    5.998_322_065_558_88e-1,
    1.369_298_809_227_358e-1,
    1.487_536_129_085_061_5e-2,
    7.868_691_311_456_133e-4,
    1.846_318_317_510_054_7e-5,
    1.421_511_758_316_446e-7,
    2.044_263_103_389_939_8e-15,
];

#[inline]
fn horner8(c: &[f64; 8], x: f64) -> f64 {
    let mut s = c[7];
    s = s * x + c[6];
    s = s * x + c[5];
    s = s * x + c[4];
    s = s * x + c[3];
    s = s * x + c[2];
    s = s * x + c[1];
    s * x + c[0]
}

/// Standard normal quantile of `p` in `(0, 1)` via Wichura's Algorithm AS 241
/// (`PPND16`). Uses `libm::log` on the 15% tail branch for cross-OS bit-exact
/// reproducibility.
#[inline]
pub fn ppnd16(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * horner8(&AS241_A, r) / horner8(&AS241_B, r);
    }
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-libm::log(tail)).sqrt();
    let v = if r <= 5.0 {
        let r = r - 1.6;
        horner8(&AS241_C, r) / horner8(&AS241_D, r)
    } else {
        let r = r - 5.0;
        horner8(&AS241_E, r) / horner8(&AS241_F, r)
    };
    if q < 0.0 { -v } else { v }
}

/// Standard normal draw from a 64-bit Philox word via `ppnd16(u01_open(w))`.
#[inline]
pub fn normal(w: u64) -> f64 {
    ppnd16(u01_open(w))
}

/// Coordinates of one model instance's Philox draw stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DrawStream {
    pub seed: u64,
    pub group_hash: u64,
    pub slot: u64,
}

impl DrawStream {
    #[inline]
    pub const fn new(seed: u64, group_hash: u64, slot: u64) -> Self {
        Self {
            seed,
            group_hash,
            slot,
        }
    }

    #[inline]
    pub const fn key(self) -> [u64; 2] {
        [self.seed, self.group_hash]
    }

    /// Block `j` of `(lane, n, slot)`.
    #[inline]
    pub fn block(self, j: u64, lane: u64, n: u64) -> [u64; 4] {
        philox4x64([j, lane, n, self.slot], self.key())
    }

    /// Word `idx` of `(lane, n, slot)` (sequential block `idx / 4`, word `idx % 4`).
    #[inline]
    pub fn word(self, idx: u64, lane: u64, n: u64) -> u64 {
        self.block(idx >> 2, lane, n)[(idx & 3) as usize]
    }

    /// Lazy 4-word buffered reader over `(lane, n, slot)` with an optional
    /// high-32-bit prefix on `j` (`j_high = (tree as u64) << 32` for forests,
    /// `0` for flat streams).
    #[inline]
    pub const fn reader(self, lane: u64, n: u64, j_high: u64) -> LaneReader {
        LaneReader {
            key: self.key(),
            lane,
            n,
            slot: self.slot,
            j_high,
            block_idx: 0,
            pos: 4,
            buf: [0; 4],
        }
    }
}

/// Allocation-free lazy buffered reader for one Philox `(lane, n, j_high)` stream.
#[derive(Debug, Clone)]
pub struct LaneReader {
    key: [u64; 2],
    lane: u64,
    n: u64,
    slot: u64,
    j_high: u64,
    block_idx: u64,
    pos: usize,
    buf: [u64; 4],
}

impl LaneReader {
    #[inline]
    pub fn next_word(&mut self) -> u64 {
        if self.pos >= 4 {
            let j = self.j_high | self.block_idx;
            self.buf = philox4x64([j, self.lane, self.n, self.slot], self.key);
            self.block_idx += 1;
            self.pos = 0;
        }
        let w = self.buf[self.pos];
        self.pos += 1;
        w
    }

    #[inline]
    pub fn next_u01(&mut self) -> f64 {
        u01(self.next_word())
    }

    #[inline]
    pub fn next_u01_open(&mut self) -> f64 {
        u01_open(self.next_word())
    }

    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        normal(self.next_word())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox4x64_matches_random123_kat_vectors() {
        // Random123 kat_vectors for Philox4x64-10: zeros, all-ones, pi digits.
        let kats: [([u64; 4], [u64; 2], [u64; 4]); 3] = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [
                    0x1655_4D9E_CA36_314C,
                    0xDB20_FE9D_672D_0FDC,
                    0xD7E7_72CE_E186_176B,
                    0x7E68_B68A_EC7B_A23B,
                ],
            ),
            (
                [u64::MAX; 4],
                [u64::MAX; 2],
                [
                    0x87B0_92C3_013F_E90B,
                    0x438C_3C67_BE8D_0224,
                    0x9CC7_D7C6_9CD7_77B6,
                    0xA09C_AEBF_594F_0BA0,
                ],
            ),
            (
                [
                    0x243F_6A88_85A3_08D3,
                    0x1319_8A2E_0370_7344,
                    0xA409_3822_299F_31D0,
                    0x082E_FA98_EC4E_6C89,
                ],
                [0x4528_21E6_38D0_1377, 0xBE54_66CF_34E9_0C6C],
                [
                    0xA528_F454_03E6_1D95,
                    0x38C7_2DBD_566E_9788,
                    0xA5A1_610E_72FD_18B5,
                    0x57BD_43B5_E52B_7FE6,
                ],
            ),
        ];
        for (ctr, key, want) in kats {
            assert_eq!(philox4x64(ctr, key), want);
        }
    }

    #[test]
    fn word_conversions_stay_in_bounds() {
        assert_eq!(u01(0), 0.0);
        let u_max = u01(u64::MAX);
        assert!(u_max < 1.0 && u_max == 1.0 - TWO_M53);

        let o_min = u01_open(0);
        let o_max = u01_open(u64::MAX);
        assert!(o_min > 0.0 && o_min == TWO_M53);
        assert!(o_max < 1.0 && o_max == 1.0 - TWO_M53);

        for k in [1usize, 2, 5, 7, 100] {
            assert_eq!(below(0, k), 0);
            assert_eq!(below(u64::MAX, k), k - 1);
        }
    }

    #[test]
    fn ppnd16_matches_exact_normal_quantiles_across_all_branches() {
        // Central branch (|p - 0.5| <= 0.425), intermediate tail (r <= 5),
        // and extreme tail (r > 5, p < exp(-25) ~ 1.38e-11).
        assert_eq!(ppnd16(0.5), 0.0);
        let cases = [
            (0.25, -0.674_489_750_196_081_7),
            (0.75, 0.674_489_750_196_081_7),
            (0.975, 1.959_963_984_540_054),
            (0.025, -1.959_963_984_540_054),
            (0.001, -3.090_232_306_167_813),
            (0.999, 3.090_232_306_167_813),
            (1e-12, -7.034_483_825_301_131),
            (1.0 - 1e-12, 7.034_486_910_047_835),
        ];
        for (p, want) in cases {
            let got = ppnd16(p);
            assert!(
                (got - want).abs() < 1e-12,
                "ppnd16({p}) = {got}, expected {want}"
            );
        }
        // Antisymmetry on exact dyadic points.
        for shift in [1u64, 17, 1023, 1 << 40] {
            let p = (shift as f64) * TWO_M53;
            let q = 1.0 - p;
            assert_eq!(ppnd16(p), -ppnd16(q));
        }
    }

    #[test]
    fn lane_reader_matches_direct_word_indexing() {
        let s = DrawStream::new(42, 0x9c36_b920_8d67_1c14, 7);
        let mut r = s.reader(LANE_POLICY_NORMAL, 13, 0);
        for i in 0..17u64 {
            assert_eq!(r.next_word(), s.word(i, LANE_POLICY_NORMAL, 13));
        }
        let tree = 5u64;
        let mut tr = s.reader(LANE_RCF_CUT, 9, tree << 32);
        for b in 0..3u64 {
            let blk = s.block((tree << 32) | b, LANE_RCF_CUT, 9);
            for w in blk {
                assert_eq!(tr.next_word(), w);
            }
        }
    }
}
