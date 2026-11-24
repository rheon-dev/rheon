//! The plan's definitions of a moments, a cross-group (its neighbor list) and a
//! profile operator, and a list column's keys: moved out of `windows_frame.rs`
//! under the source cap.

use super::*;

/// A moments operator's term on `kernel`: each leg's formula an input of
/// the core's (`input_of`), the legs' columns the distinct inputs among
/// them, and the model's settings, which the model checks here.
pub(super) fn moments_def(
    name: &str,
    kernel: &KernelDef,
    t: &crate::formula::MomentsNode,
    legs: &[Node],
    input_of: &mut dyn FnMut(&Node) -> Result<usize, Refused>,
) -> Result<MomentsDef, Refused> {
    let mut inputs: Vec<usize> = Vec::new();
    let mut powers = Vec::with_capacity(legs.len());
    for ((leg, lag), &power) in legs.iter().zip(&t.lags).zip(&t.powers) {
        let i = input_of(leg)?;
        let col = match inputs.iter().position(|&x| x == i) {
            Some(c) => c,
            None => {
                inputs.push(i);
                inputs.len() - 1
            }
        };
        powers.push((rheon_core::MomentLeg::lagged(col, lag.value()), power));
    }
    let term = rheon_core::MomentTerm {
        name: "ewm_moments".into(),
        powers,
        kind: t.kind,
        bias: t.bias(),
    };
    let cfg = MomentsDef::cfg_of(kernel, term, t.lag_tolerance.as_ref().map(Span::value));
    cfg.validate()
        .map_err(|e| format!("{WHO}: {name}: {}", e.trim_start_matches("ewm_moments: ")))?;
    Ok(MomentsDef { op: 0, inputs, cfg })
}

/// The index of a cross operator's list column of neighbor keys among the
/// plan's, checked: a list of keys, under a group column of its own.
pub(super) fn neighbor_list(
    p: &mut Plan,
    config: &WindowsConfig,
    input: &Schema,
    f: &str,
    name: &str,
    column: &str,
) -> Result<usize, Refused> {
    if config.group.is_none() {
        return Err(format!(
            "{WHO}: {f}: {name} reads the rows of the groups named in {column:?}, but the call \
             names no group column, so no row has a group to be named by. Name the group \
             column with group=."
        )
        .into());
    }
    let Some(dtype) = input.get(column) else {
        return Err(format!(
            "{WHO}: {f}: {name}'s neighbors column {column:?} is not in the input. Name a \
             column of the input that lists each row's neighbor keys."
        )
        .into());
    };
    let keys = |d: &DataType| {
        d.is_integer()
            || matches!(
                d,
                DataType::String | DataType::Categorical(..) | DataType::Enum(..)
            )
    };
    match dtype {
        DataType::List(inner) if keys(inner) => {}
        other => {
            return Err(Refused::wrong_type(format!(
                "{WHO}: {f}: {name}'s neighbors column {column:?} must be a list of group keys \
                 (integers or text), but it is {}. Give each row a list of the keys of the \
                 groups it reads, such as with pl.concat_list or a column of lists.",
                crate::arrow::dtype_name(other)
            )));
        }
    }
    Ok(match p.lists.iter().position(|c| c == column) {
        Some(i) => i,
        None => {
            p.lists.push(column.to_string());
            p.lists.len() - 1
        }
    })
}

/// A list column's keys as text, flat, and each row's range of them (`None`
/// for a null list).
pub(super) type ListKeys = (Vec<Option<String>>, Vec<Option<(usize, usize)>>);

/// [`ListKeys`] of a column.
pub(super) fn list_keys(df: &DataFrame, column: &str) -> PolarsResult<ListKeys> {
    let ca = df.column(column)?.list()?.rechunk();
    let arr = ca.downcast_iter().next().expect("rechunked");
    let values = Series::from_arrow(PlSmallStr::EMPTY, arr.values().clone())?;
    let text = crate::arrow::key_text(&values)?;
    let flat: Vec<Option<String>> = text.str()?.iter().map(|v| v.map(str::to_string)).collect();
    let offsets = arr.offsets();
    let ranges = (0..arr.len())
        .map(|i| {
            arr.validity()
                .is_none_or(|v| v.get_bit(i))
                .then(|| (offsets[i] as usize, offsets[i + 1] as usize))
        })
        .collect();
    Ok((flat, ranges))
}

/// A profile operator's definition from its node: its input and weight
/// among the plan's inputs, its spans against the clock, its zone from the
/// clock column's dtype, and its bytes against its cap.
#[allow(clippy::too_many_arguments)]
pub(super) fn profile_def(
    config: &WindowsConfig,
    input: &Schema,
    p: &mut Plan,
    f: &str,
    op: &OpNode,
    params: &crate::formula::ProfileParams,
    incr_schema: &Schema,
) -> Result<ProfileDef, Refused> {
    let name = op.kind.name();
    let resolve_incr = |op: &OpNode| col(hidden(op).as_str());
    let mut input_of = |what: &str, node: &Node| -> Result<usize, Refused> {
        if let Some(i) = p.inputs.iter().position(|n| n == node) {
            return Ok(i);
        }
        let dtype =
            dtypes_of(incr_schema, vec![node.to_expr(&resolve_incr).alias("@t")]).map_err(|e| {
                format!(
                    "{WHO}: {f}: Polars cannot compute {name}'s {what} ({e}). Check that the \
                     {what} reads columns of the right type."
                )
            })?;
        let dtype = dtype.get("@t").expect("aliased").clone();
        if !numeric(&dtype) {
            return Err(Refused::wrong_type(format!(
                "{WHO}: {f}: {name}'s {what} must be a number, but it is {}. Cast it to a \
                 number, such as with .cast(pl.Float64).",
                crate::arrow::dtype_name(&dtype)
            )));
        }
        p.inputs.push(node.clone());
        Ok(p.inputs.len() - 1)
    };
    let x = input_of("input", &op.input)?;
    let weight = match &params.weight {
        Some(w) => Some(input_of("weight", w)?),
        None => None,
    };
    let half_life = op.half_life.clone().expect("checked when read");
    p.spans.push(("half_life", half_life.clone()));
    p.spans.push(("period", params.period.clone()));
    if let Some(w) = &op.window_size {
        p.spans.push(("window_size", w.clone()));
    }
    if let Some(o) = &params.offset {
        p.spans.push(("offset", o.clone()));
    }
    let clock_dtype = config.clock.as_deref().and_then(|c| input.get(c));
    let temporal = clock_dtype.is_some_and(DataType::is_temporal);
    let phase = |s: &Span| match s {
        Span::Duration(d) => PhaseSpan::Ns(d.nanos),
        Span::Units(u) => PhaseSpan::Units(*u),
    };
    let period = phase(&params.period);
    let offset = match &params.offset {
        Some(o) => phase(o),
        None if temporal => PhaseSpan::Ns(0),
        None => PhaseSpan::Units(0.0),
    };
    let zone = match clock_dtype {
        Some(DataType::Datetime(_, Some(tz))) => {
            Some(crate::windows::canonical_zone(tz.as_str()).map_err(|_| {
                format!(
                    "{WHO}: {f}: {name} reads each row's phase on the wall clock of the clock \
                     column's time zone, but {:?} is not a zone it can read. Convert the clock \
                     to a named zone, such as with .dt.convert_time_zone(\"Europe/Amsterdam\").",
                    tz.as_str()
                )
            })?)
        }
        _ => None,
    };
    let def = ProfileDef {
        stat: params.stat,
        buckets: params.buckets,
        period,
        offset,
        zone,
        half_life: half_life.value(),
        closed: op.closed,
        input: x,
        weight,
        min_samples: op.min_samples,
        bias: op.bias,
        cap: params
            .max_state_bytes
            .unwrap_or(crate::windows::PROFILE_STATE_BYTES),
        window: op.window_size.as_ref().map(phase),
        quantile: op.quantile,
        harmonics: params.harmonics,
    };
    if def.bytes() > def.cap {
        let (what, fewer) = if def.harmonics > 0 {
            (
                format!("its fit of {} harmonics", def.harmonics),
                "harmonics",
            )
        } else {
            (
                format!(
                    "{} buckets of {} bytes each",
                    def.buckets,
                    crate::windows::PROFILE_BUCKET_BYTES
                ),
                "buckets",
            )
        };
        return Err(format!(
            "{WHO}: {f}: {name} holds {} bytes a group, {what}, past its max_state_bytes of {}. \
             Raise max_state_bytes to at least {}, or take fewer {fewer}.",
            def.bytes(),
            def.cap,
            def.bytes()
        )
        .into());
    }
    Ok(def)
}
