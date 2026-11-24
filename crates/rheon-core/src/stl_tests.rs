//! STL and MSTL against statsmodels to the bit, and the Fortran's rules one
//! by one.
//!
//! The snapshot below is statsmodels 0.15.0's output (`STL(y, ...).fit()`,
//! `MSTL(y, periods=(6, 4), windows=(15, 11), iterate=2).fit()`) on the
//! series `series` builds with no library function, printed as each
//! value's bits by
//!
//! ```text
//! PAT = [1.5, -0.5, 2.0, -3.0, 0.75, -1.25]
//! y = [((i * 37) % 11) * 0.25 + i * 0.1 + PAT[i % 6] for i in range(30)]
//! (spiked: y[7] += 25.0; y[19] -= 18.0)
//! np.asarray(component, np.float64).view(np.uint64)
//! ```
//!
//! statsmodels' cubes go through the platform's `pow`, as this port's do, so
//! the snapshot holds to the bit on the platform it was printed on (Linux
//! x86-64, glibc), and within 1e-12 of the series' scale on any other, whose
//! `pow` may round a cube's last bit differently; the Python oracles hold
//! each OS to its own statsmodels.

use super::*;

const PAT: [f64; 6] = [1.5, -0.5, 2.0, -3.0, 0.75, -1.25];

fn series(n: usize, spike: bool) -> Vec<f64> {
    let mut y: Vec<f64> = (0..n)
        .map(|i| ((i * 37) % 11) as f64 * 0.25 + i as f64 * 0.1 + PAT[i % 6])
        .collect();
    if spike {
        y[7] += 25.0;
        y[19] -= 18.0;
    }
    y
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The snapshot's platform: its values to the bit.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn same(got: &[f64], want: &[u64], what: &str) {
    assert_eq!(bits(got), want, "{what}");
}

/// Another platform: its `pow` may part from glibc's in a cube's last bit.
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn same(got: &[f64], want: &[u64], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = f64::from_bits(*w);
        assert!(
            (g - w).abs() <= 1e-12 * w.abs().max(30.0),
            "{what} row {i}: {g} against {w}"
        );
    }
}

const PLAIN_SEASONAL: [u64; 30] = [
    0x3ff10251b657ed9f,
    0xbfe083a4afa16d97,
    0x4001c934834df246,
    0xc005f024f9fcf590,
    0x3ff6472b11a72485,
    0xbfeb7aa96755feed,
    0x3ffd2198d4a265fc,
    0xc0026fb28e56503a,
    0x3ffa8dd0b7cb11f5,
    0xbff2c84874765dc2,
    0x3ff68860db731cc3,
    0xbffcea6bd97da9ab,
    0x3fff06c7822cfd89,
    0xbff8e9d34232a666,
    0x3ff234764ccec390,
    0xbff684bba7d33adf,
    0x3ffd798ade7b74b8,
    0xbff91e478e0c632e,
    0x3ff21d6b0b63908e,
    0xbff55682dda65c6e,
    0x3ffa029c6d6bf8df,
    0xbff7373d977b6273,
    0x3ff368b996b065d0,
    0xbff3e04ee4f72fd6,
    0x3ff5223d1e389524,
    0xbff5053ff979df42,
    0x3ff4907dce579390,
    0xbff2cb7ecd3efb06,
    0x3ff0415f48e54698,
    0xbff21702ef6adde6,
];
const PLAIN_TREND: [u64; 30] = [
    0x3fe86c970d6ed2c9,
    0x3fed085daf8d0256,
    0x3ff1011ff6ad74ec,
    0x3ff3d8e62c5ffb37,
    0x3ff6004276ae69eb,
    0x3ffbb7250ab832b7,
    0x40013cf293f50e4f,
    0x4001bfd58f48f422,
    0x40013fbd563b3d1b,
    0x400087c72af50bc7,
    0x3ffe2c06ae937165,
    0x400017c29fe96232,
    0x4002987701031b5d,
    0x40036e68d74e238c,
    0x4004097f2b66d233,
    0x400470cb93edd454,
    0x400529c03f6a9e6b,
    0x400833fc0283cc01,
    0x400b0e6c1e1b4032,
    0x400b4b738b3cded4,
    0x4009fd9b62fd348a,
    0x400896c4c7d57994,
    0x40080d90f9aeed3e,
    0x40096e9c917c37f5,
    0x400c4b0a08a33615,
    0x400e38f865e95c56,
    0x400ddc8255960e9d,
    0x400de8b031cedb45,
    0x400dbe381cb2d474,
    0x400d75e1774141a2,
];
const PLAIN_WEIGHTS: [u64; 30] = [
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
];
const ROBUST_SEASONAL: [u64; 30] = [
    0x3ff1c265791e8b60,
    0x3fc372cb46f60ce9,
    0x4001bf74e24ed001,
    0xc00bf9baefa4ba3a,
    0x3ff64c5da6412f04,
    0xbfa66c8c09235ff9,
    0x3ffd08267c3f9369,
    0xc0092f948f2dd5e8,
    0x3ffa5cf54c078241,
    0xbfd2734a45ebc2d5,
    0x3ff6f7c9ffb143e9,
    0xc0067b31ed561fc4,
    0x3ffe8e0f99ce15bc,
    0xbfdfcb171159ca5d,
    0x3ff2db5c33e21ad3,
    0xc0033e36796fb58e,
    0x3fff5f68a254a968,
    0xbfef9ca64c1d70a7,
    0x3ff4be7c08ea9841,
    0xc0006b5dbd5e1372,
    0x3ffc19cce3ce6577,
    0xbff376215d971323,
    0x3ff790979cadea1e,
    0xbffc59895a81c4cc,
    0x3ff6c18220223327,
    0xbff44d9537fa0227,
    0x3ff9772ef305b0b3,
    0xbff7ef77099e3395,
    0x3ff0b6263e4de493,
    0xbff4047eac1b807a,
];
const ROBUST_TREND: [u64; 30] = [
    0x3fe39c686b6cb19e,
    0x3fe921392c2773bc,
    0x3fee1069ccd098f4,
    0x3ff1759305961868,
    0x3ff344bf467488a6,
    0x3ff4cdc69028e7ec,
    0x3ffd0184daff5320,
    0x40012626393dec71,
    0x4001e3e7c7149b80,
    0x4004bb001f7f12ad,
    0x400504be20ac96ad,
    0x40035c0b48455130,
    0x40038888d7c7095d,
    0x4003c8aaf8049700,
    0x4003d8ba4a1a0bb6,
    0x4003e8cc174efc3c,
    0x400435e9f67e7a9c,
    0x400564fd8046ed55,
    0x4005d7bd5a29b308,
    0x400566cddf123b69,
    0x400528d8e5b022ab,
    0x400551d6f2f38c1b,
    0x400725f9fcec7f6b,
    0x4009a8c7becfa8b1,
    0x400c83ad52a09f3c,
    0x400e74d2c1145a16,
    0x400e19b6c76fa565,
    0x400e2f3073a3a41c,
    0x400e2a35d88e89ea,
    0x400e1a059cde0471,
];
const ROBUST_WEIGHTS: [u64; 30] = [
    0x3fefc2b43b9bb84c,
    0x3fef73e6a02300d4,
    0x3feaf7d55f504576,
    0x3feffd937830147b,
    0x3fefcf9bb8cc47c3,
    0x3fefb6c7019df274,
    0x3feb1a4cfe7c5bb4,
    0x0000000000000000,
    0x3fe70b35c2a0753c,
    0x0000000000000000,
    0x3fee7cd2fe7f5124,
    0x3fefb8c5cacc4837,
    0x3fedfb8ecd7ca168,
    0x3feccdf99dd6d3fb,
    0x3fefffe423a36787,
    0x3fef77271ecc63ab,
    0x3feff03e6aa0ea5c,
    0x3fed6ecd16610c2f,
    0x3fed2d3067a7b957,
    0x0000000000000000,
    0x3fef6a386f6e8a77,
    0x3fee4aaf2f217dc0,
    0x3fe70a45b9126c0c,
    0x3fee3397ea36382e,
    0x3fec1842fb806d1a,
    0x3fef99a38614d675,
    0x3feed2a9c56449d6,
    0x3fef7c303ef6be0b,
    0x3fed3bdd2898895c,
    0x3fee0f950bbe68cf,
];
const JUMPS_SEASONAL: [u64; 30] = [
    0x3ff6ee47c43028d6,
    0xbfefa5fd05a8f7fc,
    0x3ffb065a8cb3c490,
    0xc0017f8cf081d75f,
    0x3ffbde2a48ad320e,
    0xbff30058d51ccf2b,
    0x3ff66b199d43a8c9,
    0xc0002f692bd860a1,
    0x40009b7020384124,
    0xbff5a549b7ad95d2,
    0x3ff277d819fc5f0d,
    0xbffd590fbcf4991b,
    0x3ffe9e07f00f814e,
    0xbff57dd33c9ef8ec,
    0x3ff0adb1fa27e886,
    0xbff5c4baea58c264,
    0x3ffa7366fd33eda5,
    0xbff68bbb82c9453f,
    0x3fec152df4b86474,
    0xbfed71f801968967,
    0x3ff87bb75048a94f,
    0xbff76a7d136bbb6e,
    0x3feefc91e5f8cb63,
    0xbff08ea8b65e21bf,
    0x3ff7cafe060a1e49,
    0xbff7618dfd4c28bf,
    0x3ff17a65d673f196,
    0xbff22f123a3f9ab5,
    0x3ff68223319c61df,
    0xbff54d9f4b4e2401,
];
const JUMPS_TREND: [u64; 30] = [
    0x3fe92a9570f74077,
    0x3fee878fcd511009,
    0x3ff1f24514d56fcd,
    0x3ff4a0c243025796,
    0x3ff91ae59e995832,
    0x3ffd9508fa3058ce,
    0x400107962ae3acb5,
    0x4000d41d92eaec0c,
    0x4000a0a4faf22b62,
    0x40006d2c62f96ab9,
    0x40013607006eeb78,
    0x4001fee19de46c36,
    0x4002c7bc3b59ecf5,
    0x4003597b2cc642ba,
    0x4003eb3a1e32987e,
    0x40047cf90f9eee43,
    0x40069d5d355f6dd2,
    0x4008bdc15b1fed61,
    0x400ade2580e06cf0,
    0x400a12920c8eedc7,
    0x400946fe983d6e9e,
    0x40087b6b23ebef75,
    0x4009ca2f063b3a2d,
    0x400b18f2e88a84e6,
    0x400c67b6cad9cf9e,
    0x400cddb302472564,
    0x400d53af39b47b2a,
    0x400dc9ab7121d0f0,
    0x400d8dcffa6a049a,
    0x400d51f483b23844,
];
const JUMPS_WEIGHTS: [u64; 30] = [
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
];
const WIDE_SEASONAL: [u64; 30] = [
    0x3fdc073911341812,
    0xbf9b03deadb35b9b,
    0x400769542cd4f93d,
    0xc00eaca9451c7b26,
    0x3ff753867b15544b,
    0xbff31d2405dd856a,
    0x3fecaedd47881996,
    0xbfbc90b2cd377297,
    0x40047431144d0bbc,
    0xc00b00e17b470149,
    0x3ff17cb73eee695a,
    0xbff4a261753cf8f1,
    0x3ff5ac403b74e9ff,
    0xbfc9c29f61fb6738,
    0x400184303d48dcd6,
    0xc007555842c4a54b,
    0x3fe743bf5937c234,
    0xbff61d5f979f7eb3,
    0x3ffd01135d5404bf,
    0xbfd2d25a73cb53f3,
    0x3ffd2ce50ee00cc7,
    0xc003a9cfb7bb42e1,
    0x3fd709fe2beb7579,
    0xbff78b660abc902f,
    0x40022af7fbbadc16,
    0xbfd8ec6a7a94cde0,
    0x3ff7556f9381aa7b,
    0xbffffc11be6c8092,
    0xbf83908149a54d51,
    0xbff8e7141e05ec71,
];
const WIDE_TREND: [u64; 30] = [
    0x3ff141182913af11,
    0x3ff2f6f655c8d937,
    0x3ff4ab5e04db836e,
    0x3ff65e863ac012b3,
    0x3ff810bc3b690275,
    0x3ff9c2628419067a,
    0x3ffb7400247ba25e,
    0x3ffd25d7e7501546,
    0x3ffed8162261f754,
    0x40004569b29ca054,
    0x40011eea532ade76,
    0x4001f84e77c148d7,
    0x4002d13451b876cd,
    0x4003a93f3b7f6ec1,
    0x40047fdf7f2fd798,
    0x400553a4767d50cd,
    0x4006286dc64399e6,
    0x4006ff38d9067dd3,
    0x4007d73866eeb966,
    0x4008aff7d70abe92,
    0x40098923ad932833,
    0x400a625f0ccc27c0,
    0x400b3b6bb8e8fe5e,
    0x400c142a54a32d6a,
    0x400ceca552557e36,
    0x400dc4f20e870a48,
    0x400e9d324488be3c,
    0x400f75ac22117788,
    0x40102749ddd2f356,
    0x4010940949a5ed5b,
];
const WIDE_WEIGHTS: [u64; 30] = [
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3ff0000000000000,
];
const MSTL_SEASONAL_4: [u64; 30] = [
    0xbff89183b8556716,
    0xc018ef2a4a2b8e5a,
    0xbffe9769b6e46314,
    0x4020281f5b68cb13,
    0xbfe7db1c63534b66,
    0xc0129fca81d8a744,
    0xbff1112de935681d,
    0x401383cf1b471a80,
    0x3fb1cc8b7644c517,
    0xc0090cc1abc35eec,
    0xbfd318c5366092ea,
    0x3ffda58404868c2d,
    0x3fe9e7806399f7a4,
    0xbff9a28bfcf389d5,
    0x3fda66e3c5f05911,
    0xbff03d33d25393a3,
    0x3ff9045ec785cf91,
    0xbfdb1bcea0c1b938,
    0x3ff1db03f3697657,
    0xc00be5b929de9b91,
    0x4003536bd9b19016,
    0x3fd94d1783ad975f,
    0x3ffde16422820ced,
    0xc017b5669bce8d59,
    0x400a4ad25be2db59,
    0x3ff32e89a966df0d,
    0x4004699d51ee9bc6,
    0xc020640f6142ddb2,
    0x40103eea1447c4dc,
    0x40001dcdbee34240,
];
const MSTL_SEASONAL_6: [u64; 30] = [
    0x3ffb0ebc2d0e4dea,
    0x402657f56d24c609,
    0x4003fc8dff9ff454,
    0xc0267e8037807dfe,
    0xbfcc5d3436d241af,
    0xbffbfee1678309b9,
    0x3ff41bf49b5fce30,
    0x401a977ebbb4d91b,
    0x3fff76c490ae89cd,
    0xc01aca236c1071dd,
    0xbfd4b4591645dc22,
    0xbfe7abf517a4964a,
    0x3fe967cff96243f6,
    0x4001c887b6f12687,
    0x3ff6e929207d8266,
    0xc00118ebb2c12cb5,
    0xbfddbebf4f54056e,
    0x3fcc2eda8a2b1f99,
    0x3fd7752da5a3905b,
    0xc001a62f310bd75d,
    0x3fee93297f982232,
    0x40032622f2484849,
    0xbfe4b3806924b1dd,
    0x3ff32d2e18e470a1,
    0x3f837d4ed931fcd8,
    0xc01af52f9013e552,
    0x3fe159a1761971a7,
    0x401b7aab9eea4852,
    0xbfeb9cccaba05efb,
    0x4001dad1ee75545b,
];
const MSTL_TREND: [u64; 30] = [
    0xbff673c62ca8a1e0,
    0xbfb8996b7b5e586d,
    0x3ff35ade66387c3e,
    0x4004187889b52db9,
    0x400e88298878f457,
    0x4014fcd6d9fa2d7c,
    0x4018acb7d43356f5,
    0x401a0be07b9dda41,
    0x40188fd7ae07b9aa,
    0x4015089920a2fee8,
    0x4010deaa4533e493,
    0x400900601cb3f0dd,
    0x4003844e65d98026,
    0x4000c1e51151b0d2,
    0x4000d0f70eac3728,
    0x4000ffe70dd5322d,
    0x3ff90db6896406c3,
    0x3feadb61bb471047,
    0x3fc89fa7077c50f8,
    0xbfc22621e4a4232f,
    0xbf81b79c461faf88,
    0x3fe431ca305c5292,
    0x3ffbff05cef4c2b1,
    0x4007c691964fe3c2,
    0x400d960aa7bccfa3,
    0x400a82328af0f78c,
    0x40065bc59b5d2fd4,
    0x40019f1708c0e33c,
    0x3ff849bd157d4190,
    0x3fe7785087e3cb3c,
];
const MSTL_RESID: [u64; 30] = [
    0x4005fb46dbf7dd86,
    0xc010f7f47bca1df0,
    0x40033b51424f98b1,
    0xbffcb11d65dff752,
    0xbfdcc2dfc81e5180,
    0x40054558037e786c,
    0xc00e120634af1426,
    0x402152cf3d197f78,
    0xc007030f99b277fe,
    0x4009c30975d17808,
    0xbfb876181a5f6880,
    0xc0118dac0620a24a,
    0xbfd5f4471bf7ab90,
    0x3fb5a7f393aa7ee0,
    0xbfd2f9a723ae8248,
    0x3fe8de7a38571164,
    0x3ffef7341c84c4a1,
    0x3fd41a6ab1ead5ae,
    0x4009004447857420,
    0xc02087a094e605ea,
    0x3ff63302c55d0e1a,
    0xc0048f6ba208430c,
    0xbf94dda26c50df40,
    0x400e740afb416508,
    0xbff1824e3e8b538c,
    0x4011e873e041b1c9,
    0xbfa3bf9f8156ce00,
    0x3ff12ad1b01f3900,
    0xbfe5546487feb167,
    0xbffd4701383cac6c,
];

fn cfg(period: usize, seasonal: usize) -> StlCfg {
    StlCfg::new(period, seasonal)
}

fn check(fit: &StlFit, y: &[f64], seasonal: &[u64], trend: &[u64], weights: &[u64]) {
    same(&fit.seasonal, seasonal, "seasonal");
    same(&fit.trend, trend, "trend");
    same(&fit.weights, weights, "weights");
    let resid: Vec<f64> = (0..y.len())
        .map(|i| y[i] - fit.seasonal[i] - fit.trend[i])
        .collect();
    assert_eq!(bits(&fit.resid), bits(&resid), "resid");
}

#[test]
fn stl_is_statsmodels_to_the_bit() {
    let y = series(30, false);
    let fit = Stl::new(&cfg(4, 7)).unwrap().fit(&y);
    check(&fit, &y, &PLAIN_SEASONAL, &PLAIN_TREND, &PLAIN_WEIGHTS);

    let ys = series(30, true);
    let robust = StlCfg {
        robust: true,
        ..cfg(4, 7)
    };
    let fit = Stl::new(&robust).unwrap().fit(&ys);
    check(&fit, &ys, &ROBUST_SEASONAL, &ROBUST_TREND, &ROBUST_WEIGHTS);

    let jumps = StlCfg {
        seasonal_deg: 0,
        seasonal_jump: 2,
        trend_jump: 3,
        low_pass_jump: 2,
        ..cfg(4, 5)
    };
    let fit = Stl::new(&jumps).unwrap().fit(&y);
    check(&fit, &y, &JUMPS_SEASONAL, &JUMPS_TREND, &JUMPS_WEIGHTS);

    // Spans past the subseries' length (5 rows each): the q > n radius.
    let wide = StlCfg {
        trend: Some(41),
        low_pass: Some(7),
        ..cfg(6, 35)
    };
    let fit = Stl::new(&wide).unwrap().fit(&y);
    check(&fit, &y, &WIDE_SEASONAL, &WIDE_TREND, &WIDE_WEIGHTS);
}

#[test]
fn mstl_is_statsmodels_to_the_bit() {
    let y = series(30, true);
    let m = Mstl::new(&[(6, 15), (4, 11)], &cfg(2, 3), 2).unwrap();
    let fit = m.fit(&y);
    assert_eq!(fit.periods, vec![4, 6]);
    same(&fit.seasonal[0], &MSTL_SEASONAL_4, "seasonal 4");
    same(&fit.seasonal[1], &MSTL_SEASONAL_6, "seasonal 6");
    same(&fit.trend, &MSTL_TREND, "trend");
    same(&fit.resid, &MSTL_RESID, "resid");
}

#[test]
fn mstl_of_one_season_is_its_stl() {
    let y = series(30, true);
    let one = Mstl::new(&[(4, 7)], &cfg(2, 3), 2).unwrap().fit(&y);
    let stl = Stl::new(&cfg(4, 7)).unwrap().fit(&y);
    assert_eq!(one.seasonal[0], stl.seasonal);
    assert_eq!(one.trend, stl.trend);
}

/// A loess weight read back: degree 0 at `xs` of a unit vector at row `j`.
fn weight(n: usize, len: usize, xs: usize, l: usize, r: usize, j: usize) -> f64 {
    let mut y = vec![0.0; n];
    y[j - 1] = 1.0;
    let mut w = vec![0.0; n];
    est(&y, n, len, 0, xs, l, r, &mut w, None)
}

fn tricube(u: f64) -> f64 {
    (1.0 - u.powf(3.0)).powf(3.0)
}

#[test]
fn the_radius_past_the_series_is_the_fortrans() {
    // n = 5, span 9: h = max(xs - 1, 5 - xs) + floor((9 - 5) / 2) = 2 + 2 at xs = 3.
    let h = 4.0;
    let raw: Vec<f64> = (1..=5)
        .map(|j: usize| tricube(j.abs_diff(3) as f64 / h))
        .collect();
    let a: f64 = raw.iter().sum();
    for j in 1..=5 {
        assert_eq!(weight(5, 9, 3, 1, 5, j), raw[j - 1] / a, "row {j}");
    }
    // The paper's radius, lambda_n q / n = 2 * 9 / 5, gives other weights.
    assert_ne!(weight(5, 9, 3, 1, 5, 1), tricube(2.0 / 3.6) / a);
}

#[test]
fn the_cutoffs_are_a_thousandth_of_the_radius() {
    // 1001 rows, span 1001, xs = 501: h = 500; |j - xs| = 0.5 h -> tricube,
    // and the end rows at exactly h are > 0.999 h -> 0.
    let n = 1001;
    let w_end = weight(n, n, 501, 1, n, 1);
    assert_eq!(w_end, 0.0);
    let w_mid = weight(n, n, 501, 1, n, 501);
    let w_next = weight(n, n, 501, 1, n, 502);
    assert!(w_mid > w_next);
}

#[test]
fn the_cube_is_the_platforms_pow() {
    // Find a ratio where pow(u, 3) and u * u * u part, then check the weight
    // takes pow's: a refactor to powi or to products fails here.
    let n = 41;
    let h = 20.0;
    let mut seen = false;
    for d in 1..20usize {
        let u = d as f64 / h;
        let by_pow = tricube(u);
        let v = 1.0 - u * u * u;
        let by_mul = v * v * v;
        if by_pow != by_mul {
            let raw: Vec<f64> = (1..=n)
                .map(|j: usize| tricube(j.abs_diff(21) as f64 / h))
                .collect();
            let a: f64 = raw.iter().sum();
            assert_eq!(weight(n, n, 21, 1, n, 21 + d), by_pow / a);
            seen = true;
        }
    }
    assert!(seen, "no ratio where the two cubes part");
}

#[test]
fn degree_one_falls_back_when_the_spread_is_small() {
    // h = 0 (a single-row window): degree 1 is the weighted mean.
    let y = [3.0, 7.0];
    let mut w = vec![0.0; 2];
    assert_eq!(est(&y, 2, 1, 1, 1, 1, 1, &mut w, None), 3.0);
}

#[test]
fn a_point_whose_weights_vanish_takes_its_own_value() {
    let y = [1.0, 2.0, 4.0, 8.0, 16.0];
    let rw = [0.0; 5];
    let mut w = vec![0.0; 5];
    assert!(est(&y, 5, 3, 1, 3, 2, 4, &mut w, Some(&rw)).is_nan());
    let mut ys = vec![0.0; 5];
    let mut res = vec![0.0; 5];
    ess(&y, 5, 3, 1, 1, Some(&rw), &mut ys, &mut res);
    assert_eq!(ys, y);
}

#[test]
fn the_median_of_an_even_count_is_the_mean_of_the_middle_two() {
    // |R| = 1, 2, 3, 4: median 2.5, so h = 6 * 2.5 = 15 and rho = (1 - (r/15)^2)^2.
    let y = [1.0, 2.0, 3.0, 4.0];
    let mut rw = [0.0; 4];
    rwts(&y, &[0.0; 4], &mut rw);
    for (r, w) in [1.0, 2.0, 3.0, 4.0].iter().zip(rw) {
        let u = r / 15.0;
        let v = 1.0 - u * u;
        assert_eq!(w, v * v);
    }
}

#[test]
fn a_constant_series_has_no_season() {
    let y = vec![4.25; 48];
    let fit = Stl::new(&cfg(6, 7)).unwrap().fit(&y);
    for i in 0..48 {
        assert!(
            fit.seasonal[i].abs() < 1e-12,
            "row {i}: {}",
            fit.seasonal[i]
        );
        assert!((fit.trend[i] - 4.25).abs() < 1e-12);
    }
}

#[test]
fn the_fit_without_robustness_is_linear() {
    let stl = Stl::new(&cfg(4, 7)).unwrap();
    assert!(stl.is_linear());
    let a = series(30, false);
    let b = series(30, true);
    let mix: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 2.0 * x - 0.5 * y).collect();
    let (fa, fb, fm) = (stl.fit(&a), stl.fit(&b), stl.fit(&mix));
    for i in 0..30 {
        let s = 2.0 * fa.seasonal[i] - 0.5 * fb.seasonal[i];
        let t = 2.0 * fa.trend[i] - 0.5 * fb.trend[i];
        assert!((fm.seasonal[i] - s).abs() < 1e-12 && (fm.trend[i] - t).abs() < 1e-12);
    }
}

#[test]
fn defaults_and_refusals_are_statsmodels() {
    let stl = Stl::new(&cfg(24, 7)).unwrap();
    // ceil(1.5 * 24 / (1 - 1.5 / 7)) = ceil(45.8) = 46, made odd: 47.
    assert_eq!((stl.trend_span(), stl.low_pass_span()), (47, 25));
    let stl = Stl::new(&cfg(4, 7)).unwrap();
    assert_eq!((stl.trend_span(), stl.low_pass_span()), (9, 5));
    assert!(Stl::new(&cfg(1, 7)).unwrap_err().contains("period"));
    assert!(Stl::new(&cfg(4, 6)).unwrap_err().contains("seasonal"));
    assert!(
        Stl::new(&StlCfg {
            trend: Some(3),
            ..cfg(4, 7)
        })
        .unwrap_err()
        .contains("trend")
    );
    assert!(
        Stl::new(&StlCfg {
            low_pass: Some(3),
            ..cfg(4, 7)
        })
        .unwrap_err()
        .contains("low_pass")
    );
    assert!(
        Stl::new(&StlCfg {
            trend_deg: 2,
            ..cfg(4, 7)
        })
        .unwrap_err()
        .contains("trend_deg")
    );
    assert!(
        Stl::new(&StlCfg {
            seasonal_jump: 0,
            ..cfg(4, 7)
        })
        .unwrap_err()
        .contains("jump")
    );
    assert!(Mstl::new(&[], &cfg(4, 7), 2).is_err());
    assert!(Mstl::new(&[(4, 7)], &cfg(4, 7), 0).is_err());
    let robust = Stl::new(&StlCfg {
        robust: true,
        ..cfg(4, 7)
    })
    .unwrap();
    assert!(!robust.is_linear());
}
