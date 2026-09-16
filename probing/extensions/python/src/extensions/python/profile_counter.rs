//! Typed RecordBatch builder for `python.profile_counter`.

use std::sync::Arc;

use probing_core::core::{
    ArrayRef, DataType, Field, Int64Array, RecordBatch, Schema, SchemaRef, StringArray,
};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::Python;

use super::tbls::{try_record_batch, PythonTableError, TableResult};

fn profile_counter_schema() -> SchemaRef {
    SchemaRef::new(Schema::new(vec![
        Field::new("capture_id", DataType::Utf8, true),
        Field::new("local_step", DataType::Int64, true),
        Field::new("global_step", DataType::Int64, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("role", DataType::Utf8, true),
        Field::new("kernel_name", DataType::Utf8, true),
        Field::new("op_name", DataType::Utf8, true),
        Field::new("top_level_op", DataType::Utf8, true),
        Field::new("bottom_level_op", DataType::Utf8, true),
        Field::new("op_stack", DataType::Utf8, true),
        Field::new("calls", DataType::Int64, true),
        Field::new("duration_us", DataType::Int64, true),
        Field::new("flops", DataType::Int64, true),
        Field::new("dram_bytes", DataType::Int64, true),
        Field::new("metrics", DataType::Utf8, true),
    ]))
}

fn dict_opt_i64(dict: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<i64> {
    dict.get_item(key).ok().flatten().and_then(|v| v.extract::<i64>().ok())
}

fn dict_opt_str(dict: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<String> {
    dict.get_item(key).ok().flatten().and_then(|v| v.extract::<String>().ok())
}

pub fn profile_counter_batches() -> TableResult<Vec<RecordBatch>> {
    Python::attach(|py| {
        let schema = profile_counter_schema();
        let module = py.import("probing.profiling.torch_profiler.sql")?;
        let raw = module.call_method0("profile_counter_rows")?;
        let rows = raw.cast::<PyList>()?;
        if rows.is_empty() {
            let columns: Vec<ArrayRef> = schema
                .fields()
                .iter()
                .map(|field| {
                    if matches!(field.data_type(), DataType::Utf8) {
                        Arc::new(StringArray::from(Vec::<Option<String>>::new())) as ArrayRef
                    } else {
                        Arc::new(Int64Array::from(Vec::<Option<i64>>::new())) as ArrayRef
                    }
                })
                .collect();
            return Ok(vec![try_record_batch(schema, columns)?]);
        }

        let mut capture_id = Vec::new();
        let mut local_step = Vec::new();
        let mut global_step = Vec::new();
        let mut rank = Vec::new();
        let mut role = Vec::new();
        let mut kernel_name = Vec::new();
        let mut op_name = Vec::new();
        let mut top_level_op = Vec::new();
        let mut bottom_level_op = Vec::new();
        let mut op_stack = Vec::new();
        let mut calls = Vec::new();
        let mut duration_us = Vec::new();
        let mut flops = Vec::new();
        let mut dram_bytes = Vec::new();
        let mut metrics = Vec::new();

        for item in rows.iter() {
            let dict = item.cast::<PyDict>().map_err(|_| {
                PythonTableError::BatchBuild("profile_counter row not dict".into())
            })?;
            capture_id.push(dict_opt_str(&dict, "capture_id"));
            local_step.push(dict_opt_i64(&dict, "local_step"));
            global_step.push(dict_opt_i64(&dict, "global_step"));
            rank.push(dict_opt_i64(&dict, "rank"));
            role.push(dict_opt_str(&dict, "role"));
            kernel_name.push(dict_opt_str(&dict, "kernel_name"));
            op_name.push(dict_opt_str(&dict, "op_name"));
            top_level_op.push(dict_opt_str(&dict, "top_level_op"));
            bottom_level_op.push(dict_opt_str(&dict, "bottom_level_op"));
            op_stack.push(dict_opt_str(&dict, "op_stack"));
            calls.push(dict_opt_i64(&dict, "calls"));
            duration_us.push(dict_opt_i64(&dict, "duration_us"));
            flops.push(dict_opt_i64(&dict, "flops"));
            dram_bytes.push(dict_opt_i64(&dict, "dram_bytes"));
            metrics.push(dict_opt_str(&dict, "metrics"));
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(capture_id)),
            Arc::new(Int64Array::from(local_step)),
            Arc::new(Int64Array::from(global_step)),
            Arc::new(Int64Array::from(rank)),
            Arc::new(StringArray::from(role)),
            Arc::new(StringArray::from(kernel_name)),
            Arc::new(StringArray::from(op_name)),
            Arc::new(StringArray::from(top_level_op)),
            Arc::new(StringArray::from(bottom_level_op)),
            Arc::new(StringArray::from(op_stack)),
            Arc::new(Int64Array::from(calls)),
            Arc::new(Int64Array::from(duration_us)),
            Arc::new(Int64Array::from(flops)),
            Arc::new(Int64Array::from(dram_bytes)),
            Arc::new(StringArray::from(metrics)),
        ];
        Ok(vec![try_record_batch(schema, columns)?])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_counter_schema_has_expected_columns() {
        let schema = profile_counter_schema();
        for col in [
            "capture_id",
            "kernel_name",
            "op_name",
            "flops",
            "dram_bytes",
            "metrics",
        ] {
            assert!(schema.field_with_name(col).is_ok(), "missing column {col}");
        }
    }
}
