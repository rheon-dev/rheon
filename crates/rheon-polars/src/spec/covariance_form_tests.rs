//! `kalman`'s `covariance_form` (docs/PLAN.md task 281 (3)): the spec's
//! word, from JSON and from a TOML table, builds the filter in that form;
//! another word is refused naming the two; and a spec at the default
//! writes no field, so a spec written before it serializes as it did.

use super::Spec;
use crate::stream::{AnyModel, build_models};
use rheon_core::CovarianceForm;

fn json(model: &str) -> Result<Spec, String> {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"],
            "features": ["x"], "half_life": 10}}"#
    ))
    .map_err(|e| e.to_string())
}

fn form(spec: &Spec) -> CovarianceForm {
    match &build_models(spec).unwrap()[0].1 {
        AnyModel::Kalman(m) => m.covariance_form(),
        _ => panic!("not a kalman"),
    }
}

#[test]
fn a_spec_builds_the_covariance_form_unless_it_names_the_square_root() {
    for (model, want) in [
        (
            r#"{"type": "kalman", "coef_half_life": 50}"#,
            CovarianceForm::Covariance,
        ),
        (
            r#"{"type": "kalman", "coef_half_life": 50, "covariance_form": "covariance"}"#,
            CovarianceForm::Covariance,
        ),
        (
            r#"{"type": "kalman", "coef_half_life": 50, "covariance_form": "square_root"}"#,
            CovarianceForm::SquareRoot,
        ),
    ] {
        let mut s = json(model).unwrap();
        s.check().unwrap();
        assert_eq!(form(&s), want, "{model}");
        let back = serde_json::to_string(&s.model).unwrap();
        assert_eq!(
            back.contains("covariance_form"),
            want == CovarianceForm::SquareRoot,
            "{back}"
        );
    }
    let table: Spec = toml::from_str(
        r#"name = "m"
           targets = ["y"]
           features = ["x"]
           half_life = 10.0
           [model]
           type = "kalman"
           coef_half_life = 50.0
           covariance_form = "square_root""#,
    )
    .unwrap();
    assert_eq!(form(&table), CovarianceForm::SquareRoot);
}

/// serde refuses another word naming the two, and every surface words it
/// as the spec's other words are worded (`word_refusal_body`).
#[test]
fn another_word_is_refused_naming_the_two() {
    let e = json(r#"{"type": "kalman", "coef_half_life": 50, "covariance_form": "factor"}"#)
        .unwrap_err();
    let msg = e.split(" at line ").next().unwrap();
    assert_eq!(
        super::word_refusal_body(msg).as_deref(),
        Some(
            "must be \"covariance\" or \"square_root\", but it is \"factor\". Use one of \
             those words."
        ),
        "{e}"
    );
}
