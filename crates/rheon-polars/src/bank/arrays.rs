//! The output arrays that are not one `F64Column` each, built by hand, and
//! the naming of each spec's struct array as a polars column. Moved out of
//! `bank.rs` to keep that file under the repository's per-file size cap.

use super::*;

/// The three output fields that do not come from an [`F64Column`], as Arrow
/// arrays: the drift flag, the selected slot's name, and the selected and
/// averaged predictions (docs/PLAN.md task 86).
pub(super) fn opt_f64_array(vals: &[Option<f64>]) -> Box<dyn ArrowArray> {
    let values: Vec<f64> = vals.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let validity = MutableBitmap::from_trusted_len_iter(vals.iter().map(Option::is_some));
    let validity = (validity.unset_bits() > 0).then(|| validity.into());
    Box::new(PrimitiveArray::new(
        ArrowDataType::Float64,
        values.into(),
        validity,
    ))
}

pub(super) fn opt_bool_array(vals: &[Option<bool>]) -> Box<dyn ArrowArray> {
    let values = MutableBitmap::from_trusted_len_iter(vals.iter().map(|v| v.unwrap_or(false)));
    let validity = MutableBitmap::from_trusted_len_iter(vals.iter().map(Option::is_some));
    let validity = (validity.unset_bits() > 0).then(|| validity.into());
    Box::new(BooleanArray::new(
        ArrowDataType::Boolean,
        values.into(),
        validity,
    ))
}

/// The `coef` field: a list per row, null where the row carried none. Built by
/// hand rather than through a builder because the nulls are nested -- a null
/// list and a null *inside* a list are different things, and the finite-or-null
/// contract applies to both.
pub(super) fn coef_list_array(coef: &[Option<&Vec<f64>>]) -> Box<dyn ArrowArray> {
    let mut offsets: Vec<i64> = Vec::with_capacity(coef.len() + 1);
    offsets.push(0);
    let mut values: Vec<f64> = Vec::new();
    let mut inner = MutableBitmap::new();
    let mut outer = MutableBitmap::with_capacity(coef.len());
    for v in coef {
        match v {
            Some(flat) => {
                for &c in flat.iter() {
                    let ok = c.is_finite();
                    values.push(if ok { c } else { f64::NAN });
                    inner.push(ok);
                }
                outer.push(true);
            }
            None => outer.push(false),
        }
        offsets.push(values.len() as i64);
    }
    let inner_validity = (inner.unset_bits() > 0).then(|| inner.into());
    let child = PrimitiveArray::new(ArrowDataType::Float64, values.into(), inner_validity);
    let dtype = ListArray::<i64>::default_datatype(ArrowDataType::Float64);
    let outer_validity = (outer.unset_bits() > 0).then(|| outer.into());
    Box::new(ListArray::<i64>::new(
        dtype,
        OffsetsBuffer::try_from(offsets).expect("offsets are monotonic by construction"),
        Box::new(child),
        outer_validity,
    ))
}

/// Nested coefficient lists (`List(List(Float64))`) for fixed-lag smoothed
/// `kalman` outputs under delayed learning: `None` on rows with zero emitting
/// releases (materialized as `null`, never `[]`), and one inner list per
/// emitting release in release order.
pub(super) fn nested_coef_list_array(rows: &[Option<&Vec<Vec<f64>>>]) -> Box<dyn ArrowArray> {
    let mut offsets: Vec<i64> = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    let mut outer = MutableBitmap::with_capacity(rows.len());
    let mut inner_rows: Vec<Option<&Vec<f64>>> = Vec::new();
    for row in rows {
        match row {
            Some(emissions) => {
                for e in *emissions {
                    inner_rows.push(Some(e));
                }
                outer.push(true);
            }
            None => outer.push(false),
        }
        offsets.push(inner_rows.len() as i64);
    }
    let child = coef_list_array(&inner_rows);
    let dtype = ListArray::<i64>::default_datatype(child.dtype().clone());
    let outer_validity = (outer.unset_bits() > 0).then(|| outer.into());
    Box::new(ListArray::<i64>::new(
        dtype,
        OffsetsBuffer::try_from(offsets).expect("offsets are monotonic by construction"),
        child,
        outer_validity,
    ))
}

/// Clock-typed lists (`List(<clock_dtype>)`) for `smoothed_for` under delayed
/// learning: `None` on rows with zero emitting releases (materialized as
/// `null`, never `[]`), preserving the exact clock dtype and inner nulls.
pub(super) fn clock_list_array(
    clock: &crate::stream::ClockDtype,
    rows: &[Option<&Vec<Option<ClockValue>>>],
) -> Box<dyn ArrowArray> {
    let mut offsets: Vec<i64> = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    let mut flat: Vec<Option<ClockValue>> = Vec::new();
    let mut outer = MutableBitmap::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(clocks) => {
                flat.extend_from_slice(clocks);
                outer.push(true);
            }
            None => outer.push(false),
        }
        offsets.push(flat.len() as i64);
    }
    let child = clock.array(&flat);
    let dtype = ListArray::<i64>::default_datatype(child.dtype().clone());
    let outer_validity = (outer.unset_bits() > 0).then(|| outer.into());
    Box::new(ListArray::<i64>::new(
        dtype,
        OffsetsBuffer::try_from(offsets).expect("offsets are monotonic by construction"),
        child,
        outer_validity,
    ))
}

/// One spec's struct array as a named polars column. Naming the struct is
/// the whole of what the polars adapter adds to an Arrow output.
pub(super) fn named_column(spec: &Spec, st: StructArray) -> PolarsResult<Column> {
    Ok(Series::from_arrow(spec.name.as_str().into(), Box::new(st))?.into())
}

/// Every spec's struct array, named, in spec order.
pub(super) fn named_columns(specs: &[Spec], arrays: Vec<StructArray>) -> PolarsResult<Vec<Column>> {
    debug_assert_eq!(specs.len(), arrays.len(), "one struct array per spec");
    specs
        .iter()
        .zip(arrays)
        .map(|(s, st)| named_column(s, st))
        .collect()
}
