<!-- kind: record -->
# Stream processors rheon could run inside or beside, 2026-11-20

*Status, 2026-11-20: a record, in progress. The research for PLAN task 531
(request A8): which stream processors rheon should support beyond the Kafka
consumer of task 530, and what each would cost. Read-only: nothing here was
built or measured. Part 1 covers the first six candidates and part 2 the next
six; the rest, the ranking and the proposed tasks follow in later parts.*

## The question

Task 530 runs rheon as a Kafka consumer: the topic is partitioned by the group
key, a consumer reads a batch, runs the bank on it as an Arrow/Polars frame,
writes predictions, and checkpoints `save_state` with the partitions'
offsets, state first; on replay `ModelBank.skip_learned` drops rows at or
before each group's last clock, so a crash between the two learns nothing
twice. Task 531 asks, for each other stream processor: how rheon would run
inside or beside it, and what that would cost.

What rheon needs from a host, read from task 530 and the bank's API as of
`origin/main` on 2026-11-20:

- **Groups keyed per partition.** Every row of a group must reach the same
  bank, so the host must route by the group key (or by a key that a group
  never crosses) to a stable owner.
- **Rows ordered within a group.** Predictions are out-of-sample in row order
  and a step back on the clock is a stated event, so the host must deliver a
  group's rows in their order, or deliver enough (an offset or clock column)
  for rheon to restore it within a batch.
- **State that is a checkpointable value.** A bank saves to bytes
  (`save_bytes`) and loads from them (`load_bytes`); it is one value for all
  its groups. The bank can drop groups (`drop_groups`) and report each
  group's last clock (`last_clocks`), but has no call today that saves or
  loads one group's state alone.
- **State and offsets committed atomically**, or else an ordering of the two
  plus `skip_learned` that makes replay learn nothing twice.
- **An Arrow or Python path.** The bank's input is a Polars frame (Arrow
  memory); its cheapest entry is a batch of many rows, and a per-row call
  costs the Python and Arrow round trip each time.
- **Hard rule 12.** No new library statically linked into the wheel or the
  CLI without raising it first.

Scope: the projects' own documentation as read on 2026-11-20. Where a page
could not be read, or did not state the point, the text says "not verified".
Adoption is given from the projects' own pages and package indexes, not from
surveys.

## The criteria

### (a) Keying and ordering per partition

Does the host route rows by a key to a stable owner, and keep their order
within that key, from the source partition to the function that runs rheon?

### (b) State and checkpoints

Where a rheon state (the bank's bytes) could live in the host's own state or
checkpoint, or be pointed at from it; and whether the state and the source
offsets can be committed atomically.

### (c) Arrow or Python path, per-batch overhead

How rows reach Python (or another language rheon can serve), in what batch,
and what each call costs.

### (d) Replay and exactly-once

What the host replays after a failure, and which guarantee it gives for state
and for output.

### (e) Statically linked native dependencies

Whether running rheon there would link anything new into rheon's own wheel or
binary (hard rule 12), and the host's own native pieces for context.

### (f) Adoption

How widely used and how actively maintained the project is.

## 1. Apache Flink, with PyFlink and its Arrow Python functions

**(a) Keying and ordering.** `key_by` turns a stream into a `KeyedStream`
that partitions records by a deterministic key, and keyed state is scoped to
the key of the current element [F1]. Across a redistributing exchange such as
`keyBy`, "the ordering among the elements is only preserved within each pair
of sending and receiving subtasks" [F2]. A Kafka partition is read by one
source subtask, so a group whose rows all sit in one Kafka partition keeps
their order after `key_by` on the group key; rows of one group spread over
several partitions would not.

**(b) State and checkpoints.** A checkpoint holds "a pointer into each of the
data sources (e.g., an offset into a file or Kafka partition)" and a copy of
every stateful operator's state as of exactly those positions [F3]; the Kafka
source keeps its offsets in that checkpoint and "does NOT rely on committed
offsets for fault tolerance" [F4]. So a rheon state held in Flink state is
committed atomically with the offsets by construction, the strongest of the
six. The fit is the granularity: keyed state is one value per key [F1], and
a rheon bank is one value for all its groups. Keyed by group, each key's
`ValueState` would hold one group's state, which rheon cannot yet save or
load alone, and each element would pay a load and a save. Keyed by a shard
(the Kafka partition, say), one value holds a whole bank, saved at each
update unless the function can save only at snapshot time; whether PyFlink
exposes a snapshot hook for operator state was not verified. Savepoints
carry the same state across job upgrades, scoped by operator `uid` [F5].

**(c) Arrow or Python path.** Python functions run in a separate worker
process by default, talking to the JVM over gRPC; a `thread` mode embeds
Python in the JVM through PEMJA but supports no pandas UDFs or UDAFs [F6].
Vectorized ("pandas") UDFs move data "in batches using Arrow's columnar
format" [F7], with `python.fn-execution.arrow.batch.size` defaulting to 1,000
elements and the general bundle to 1,000 elements or 1,000 ms [F8]. But the
vectorized functions are stateless scalars or window aggregates whose whole
group or window is loaded into memory [F7]; the stateful path,
`KeyedProcessFunction` with `ValueState` in the DataStream API [F9], is
called per element. No documented function combines keyed state with Arrow
batches (not verified beyond the pages cited). So rheon inside PyFlink runs a
row at a time, or runs on a window's batch without carrying state from one
window to the next.

**(d) Replay and exactly-once.** Flink recovers "by rewinding and replaying
the source data streams" to the checkpoint's positions, and every event
affects Flink-managed state exactly once; end-to-end needs replayable sources
and transactional or idempotent sinks [F3]. The Kafka sink in `EXACTLY_ONCE`
writes in a Kafka transaction committed on the checkpoint, so output is
visible only after it [F4].

**(e) Static linking.** Rheon would run as its own wheel inside the Python
worker, linking nothing new. PyFlink itself requires `pemja` (not on
Windows), `apache-beam >=2.69,<=2.75`, `pyarrow <26` and `pandas <2.3` [F10],
and lists Python 3.9 to 3.12 [F11], so a rheon user on PyFlink is held to
Python 3.12 and those pins. `thread` mode, needing PEMJA, is therefore not
available on Windows.

**(f) Adoption.** An Apache top-level project; its "Powered By" page lists 33
organizations, including Alibaba, AWS, Capital One and Comcast [F12]. PyFlink
2.3.0 was released on 2026-06-21 [F11].

Sources:
[F1] https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/state/
[F2] https://nightlies.apache.org/flink/flink-docs-stable/docs/learn-flink/overview/
[F3] https://nightlies.apache.org/flink/flink-docs-stable/docs/learn-flink/fault_tolerance/
[F4] https://nightlies.apache.org/flink/flink-docs-stable/docs/connectors/datastream/kafka/
[F5] https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/state/savepoints/
[F6] https://nightlies.apache.org/flink/flink-docs-master/api/python/user_guide/execution_mode.html
[F7] https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/python/table/udfs/vectorized_python_udfs/
[F8] https://nightlies.apache.org/flink/flink-docs-release-2.3/api/python/user_guide/configuration.html
[F9] https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/operators/process_function
[F10] https://raw.githubusercontent.com/apache/flink/master/flink-python/setup.py
[F11] https://pypi.org/project/apache-flink/
[F12] https://flink.apache.org/what-is-flink/powered-by/

## 2. Kafka Streams

**(a) Keying and ordering.** A stream partition is "a totally ordered
sequence of data records and maps to a Kafka topic partition"; the library
creates a fixed number of tasks from the input partitions, the assignment of
partitions to tasks never changes, and tasks process records one at a time
[K1]. Partitioning the topic by the group key gives rheon exactly its two
needs, with no exchange in between.

**(b) State and checkpoints.** Each task embeds local state stores, each
backed by a replicated changelog topic from which it is restored after a
failure, with optional standby replicas [K1]. Persistent key-value stores are
RocksDB by default [K3]. A store keyed by group, or one entry per task
holding the bank, could hold rheon's bytes. Under exactly-once v2, "updates
on the state stores, and writes to the output topics will be completed
atomically" with the input offsets [K2], so state, predictions and offsets
commit together without rheon's state-then-offset ordering.

**(c) Arrow or Python path.** None: Kafka Streams is a Java library, and the
documentation read describes no Python API [K1, K2]. Rheon would run either
beside it as a service (a call per record or batch over the network, which
breaks the atomicity of (b) unless the state stays in the store and is sent
with each call), or inside the JVM through a native binding to rheon's Rust
core, a new interface not designed here.

**(d) Replay and exactly-once.** Exactly-once processing "guarantees each
record is processed once and only once" across client or broker failures,
for state stores and output topics [K2]; it needs brokers 2.5 or newer, and
the first version is deprecated since 3.0 [K2].

**(e) Static linking.** Running beside it as a service links nothing into
rheon. A JNI binding would ship rheon's Rust core as a native library loaded
by the JVM, which is a new deployment artifact and a linking question to
raise under rules 11 and 12 before any build. The library's RocksDB comes
from its JNI jar (whether that is statically linked: not verified).

**(f) Adoption.** Kafka is used by "over 80% of the Fortune 100"; the
"Powered By" page names Kafka Streams users including The New York Times,
Pinterest, Salesforce and Rabobank [K4].

Sources:
[K1] https://kafka.apache.org/44/streams/architecture/
[K2] https://kafka.apache.org/44/streams/core-concepts/
[K3] https://kafka.apache.org/43/streams/developer-guide/config-streams/ and https://kafka.apache.org/44/streams/developer-guide/memory-mgmt/
[K4] https://kafka.apache.org/powered-by/

## 3. Spark Structured Streaming

**(a) Keying and ordering.** The Kafka source exposes `partition` and
`offset` columns on every row [S1], and `groupBy(key)` followed by a stateful
operator hands each key's rows of a micro-batch to one call [S2]. Neither the
`transformWithState` guide nor the `applyInPandasWithState` and
`transformWithStateInPandas` docstrings states any order of rows within a
group [S2, S3]; the grouping is a shuffle. Rheon would sort each group's rows
by (partition, offset) or by its clock inside the call, which the offset
column makes possible within a micro-batch.

**(b) State and checkpoints.** A query's checkpoint location holds "the range
of offsets processed in each trigger" and the running state [S4], and the
state store is updated "in sets transactionally", each set a new version
[S4], so state and offsets advance together per micro-batch. Since Spark 4.0,
`transformWithStateInPandas` gives a processor `ValueState`, `ListState` and
`MapState` with a schema [S2]; the older `applyInPandasWithState` holds a
tuple per key [S3]. A binary field could hold a group's rheon state (one
group per key, which rheon cannot yet save alone) or a shard's bank. Whether
`transformWithState` requires the RocksDB state store in open-source Spark
was not verified; Databricks' documentation configures it [S5].

**(c) Arrow or Python path.** Data reaches pandas through Arrow, in record
batches of at most `spark.sql.execution.arrow.maxRecordsPerBatch` rows
(default 10,000) [S6]; for grouped functions "all data for a group will be
loaded into memory" and the batch limit does not apply [S6], while the state
APIs pass an iterator of pandas frames [S3]. The guide prefers the Row
interface over pandas when keys are many, as pandas conversion costs "are
considerably higher" [S2]. Each micro-batch therefore costs a shuffle, an
Arrow-to-pandas conversion, and pandas-to-Polars on rheon's side, plus the
micro-batch latency.

**(d) Replay and exactly-once.** Replayable sources (Kafka is one) are
re-read from the checkpointed offsets [S4]. The Kafka sink is at-least-once,
and Spark "cannot prevent such duplicates" [S1]; `foreachBatch` is
at-least-once unless the writer deduplicates by `batchId` [S4]. Learning is
exactly-once through the versioned state; predictions written to Kafka may
repeat.

**(e) Static linking.** Rheon would run as its own wheel in the Python
workers, linking nothing new. Spark is a JVM system driven from Python.

**(f) Adoption.** Very wide; the "Powered By" page lists Amazon, Alibaba,
Tencent, Yahoo! and Databricks among others [S7]. The PySpark stateful APIs
used here are new in 4.0 and are not supported in the 4.3 real-time mode
[S2].

Sources:
[S1] https://spark.apache.org/docs/latest/streaming/structured-streaming-kafka-integration.html
[S2] https://spark.apache.org/docs/latest/streaming/structured-streaming-transform-with-state.html
[S3] https://raw.githubusercontent.com/apache/spark/master/python/pyspark/sql/pandas/group_ops.py
[S4] https://spark.apache.org/docs/latest/streaming/apis-on-dataframes-and-datasets.html
[S5] https://docs.databricks.com/aws/stateful-applications
[S6] https://spark.apache.org/docs/latest/api/python/tutorial/sql/arrow_pandas.html
[S7] https://spark.apache.org/powered-by.html

## 4. Bytewax

The documentation site (docs.bytewax.io) did not resolve from this session;
the same guides were read from the project's repository.

**(a) Keying and ordering.** `key_on` keys a stream, and Bytewax exchanges
items between workers before stateful steps [B1]. Kafka partitions are
assigned to workers, and partitions are the unit of parallelism [B2]. Whether
a key always reaches the same worker, and whether a key's items keep their
order through the exchange, is not stated in the pages read: not verified.

**(b) State and checkpoints.** A stateful operator's logic returns its state
from `snapshot`, which must be picklable and "effectively immutable" [B1], so
rheon's `save_bytes` fits it directly. Snapshots are taken at a snapshot
interval into SQLite recovery partitions, and the dataflow resumes only from
a consistent snapshot across all partitions [B3]; with recovery on, Kafka
offsets live in those recovery partitions, not in consumer groups [B2]. So
state and input positions are snapshotted together. The state is per key, so
the granularity question of Flink applies here too.

**(c) Arrow or Python path.** Operators are Python; `stateful_batch` gives
the lowest-level control, with `on_batch` receiving items from upstream [B1]
(the docstrings disagree on whether it gets all queued items at once). Items
moved between workers are pickled [B4]. No Arrow path is documented in the
pages read; rheon would build a frame from each batch's Python objects.

**(d) Replay and exactly-once.** At-least-once: after a resume, output
systems may "see duplicate data around resume" [B3], and the Kafka guide
describes at-least-once delivery with recovery on [B2]. Learning would be
exactly-once only if snapshot and offsets are consistent as described.

**(e) Static linking.** Rheon would run as its own wheel; Bytewax ships its
Rust engine in its own wheel [B5], and its Kafka connector uses
`confluent_kafka` [B2]. Nothing new in rheon.

**(f) Adoption.** About 2.1k stars; the README says it "is now
community-maintained", the company having stopped in May 2025 [B5]. The
latest PyPI release is 0.21.1 of 2024-11-25, classified Beta, Python 3.8 to
3.12 [B6]. A weak base to build on.

Sources:
[B1] https://raw.githubusercontent.com/bytewax/bytewax/main/pysrc/bytewax/operators/__init__.py
[B2] https://raw.githubusercontent.com/bytewax/bytewax/main/docs/guide/concepts/kafka.md
[B3] https://raw.githubusercontent.com/bytewax/bytewax/main/docs/guide/concepts/recovery.md
[B4] https://raw.githubusercontent.com/bytewax/bytewax/main/docs/guide/concepts/workers-parallelization.md
[B5] https://github.com/bytewax/bytewax
[B6] https://pypi.org/project/bytewax/

## 5. Quix Streams

**(a) Keying and ordering.** State is per message key and per partition,
each partition with its own RocksDB instance [Q1]. Messages with the same key
land on the same partition, and "message ordering is only guaranteed per
key"; `group_by` re-keys through an internal repartition topic [Q2]. A topic
keyed by group gives rheon both needs.

**(b) State and checkpoints.** State is backed by compacted changelog topics
with the source's partition count, and rebuilt from them on start or
rebalance [Q1]. Values are JSON by default, or pickle, or raw bytes through
`get_bytes`/`set_bytes` [Q1], which fits rheon's `save_bytes`. A checkpoint
flushes sinks, produces the changelog, flushes the producer, commits the
offsets, then flushes the local stores [Q3]; if the changelog fails, offsets
are not committed and the store is not flushed [Q3].

**(c) Arrow or Python path.** A `StreamingDataFrame` processes each message
[Q4]; no Arrow or pandas path is documented. Batching exists for sinks, which
accumulate records and write them at the checkpoint's flush [Q5]. Rheon would
either run per message (a frame of one row, with the group's state read and
written each time) or be a batching sink that runs the bank at each flush,
with state kept beside Quix rather than in its store.

**(d) Replay and exactly-once.** At-least-once by default; `exactly-once`
holds produced messages until the checkpoint completes, at most about 10%
slower by the project's own estimate [Q6]. Whether this uses Kafka
transactions is not stated on the pages read: not verified.

**(e) Static linking.** Pure Python on top of `confluent-kafka` and
`rocksdict` [Q7], both their own wheels; nothing new in rheon.

**(f) Adoption.** About 1.6k stars [Q4]; actively released (3.28.1 on
2026-10-15), Python 3.11 or newer, Apache-2.0 [Q8].

Sources:
[Q1] https://quix.io/docs/quix-streams/advanced/stateful-processing.html
[Q2] https://quix.io/docs/quix-streams/groupby.html
[Q3] https://quix.io/docs/quix-streams/advanced/checkpointing.html
[Q4] https://github.com/quixio/quix-streams
[Q5] https://quix.io/docs/quix-streams/connectors/sinks/custom-sinks.html
[Q6] https://quix.io/docs/quix-streams/configuration.html
[Q7] https://raw.githubusercontent.com/quixio/quix-streams/main/requirements.txt
[Q8] https://pypi.org/project/quixstreams/ and https://pypi.org/rss/project/quixstreams/releases.xml

## 6. Faust (faust-streaming)

**(a) Keying and ordering.** Tables are sharded by partition and must share
shards with their source stream; a different key needs `group_by`, which
re-keys through an intermediate topic [A1, A2]. Whether an agent with
concurrency above one keeps a partition's order was not stated on the pages
read: not verified.

**(b) State and checkpoints.** Each table has a compacted changelog topic and
is rebuilt from it; RocksDB is the production store, in-memory the default
[A1, A3]. Values are JSON by default, or bytes [A1], which fits rheon's
bytes. An abrupt stop can write changelog entries before the source offsets
are committed, double-counting, which `processing_guarantee` addresses [A1].

**(c) Arrow or Python path.** Async Python agents iterate events;
`Stream.take(max_, within)` gives batches of up to N events or what arrived
within a timeout [A2]. No Arrow path; rheon would build a frame from each
batch of Python objects.

**(d) Replay and exactly-once.** At-least-once by default; `exactly_once`
configures consumers with `read_committed` and idempotent producers, and
needs three brokers by default [A3]. Offsets are committed by a background
task every few seconds by default [A2].

**(e) Static linking.** Built on `aiokafka`; RocksDB through optional extras
[A4]. Nothing new in rheon.

**(f) Adoption.** A community fork of Robinhood's Faust, about 1.9k stars
[A5]; 0.15.4 released 2026-10-28, Python 3.10 or newer [A4, A6].

Sources:
[A1] https://faust-streaming.github.io/faust/userguide/tables.html
[A2] https://faust-streaming.github.io/faust/userguide/streams.html
[A3] https://faust-streaming.github.io/faust/userguide/settings.html
[A4] https://pypi.org/pypi/faust-streaming/json
[A5] https://github.com/faust-streaming/faust
[A6] https://pypi.org/rss/project/faust-streaming/releases.xml

## 7. Redpanda

Redpanda is a broker, not a processor: it speaks the Kafka API, so the
question is whether task 530's consumer runs on it unchanged.

**(a) Keying and ordering.** Kafka clients for protocol 0.11 or later work
with it, and `confluent-kafka-python` and `librdkafka` are among the clients
it validates [R1]. Partitions, keys and per-partition order are Kafka's, so
the consumer of task 530 gets the same keying and order (that the partition
order matches Kafka's in every case was not tested; the claim is the
compatibility claim of [R1]).

**(b) State and checkpoints.** No state of its own for a consumer. Rheon's
state stays where task 530 puts it. Transactions are enabled by default, and
a consume-transform-produce loop can send its offsets in the same transaction
as its output (`sendOffsetsToTransaction`) [R2]; rheon's state file is still
outside that transaction, so the state-then-offset order plus
`skip_learned` is what makes learning exactly-once, as on Kafka.

**(c) Arrow or Python path.** Through a Kafka client, exactly as task 530.
Redpanda's own in-broker data transforms run as WebAssembly, in JavaScript,
Go or Rust, a single record at a time, with no aggregations or joins and
at-least-once delivery [R3]; they cannot hold a bank, so rheon has no place
inside the broker.

**(d) Replay and exactly-once.** Exactly-once with idempotent producers and
transactions [R2]; atomicity is not guaranteed under remote recovery, a stuck
transaction can block `read_committed` consumers, and KIP-890 is not
implemented, so Kafka 4.x clients fall back to the older transaction protocol
[R1, R2].

**(e) Static linking.** None new for the Python adapter. A Rust CLI consumer
would meet the same `librdkafka` question task 530 raises.

**(f) Adoption.** About 12.7k stars; the README calls it "Kafka API
compatible ... No ZooKeeper. No JVM!" [R4]. That GitHub page now says the
repository is archived and read-only, with issues moved to
`redpanda-data/community`, and the Community Edition is under Redpanda's own
licence [R4]; where development continues was not verified.

Sources:
[R1] https://docs.redpanda.com/current/develop/kafka-clients/
[R2] https://docs.redpanda.com/current/develop/transactions/
[R3] https://docs.redpanda.com/current/develop/data-transforms/how-transforms-work/
[R4] https://github.com/redpanda-data/redpanda

## 8. Apache Pulsar

**(a) Keying and ordering.** With a message key, the producer hashes it to a
partition, so messages of one key stay in order; there is no order across
partitions [P1]. Of the subscription types, Exclusive and Failover give one
consumer per partition (Failover can duplicate or reorder during a switch),
Shared gives no order, and Key_Shared delivers all messages of a key to one
consumer and keeps their order, if producers batch by key or not at all [P1].

**(b) State and checkpoints.** A consumer's position is a cursor per
subscription, stored durably in BookKeeper [P1, P2]. Transactions let an
application "consume, process, and produce messages in one atomic operation"
across partitions [P3], so acknowledgements and output commit together; a
rheon state file would again sit outside, behind the same state-then-ack
order as task 530. Pulsar Functions keep state in BookKeeper's table service
through `put_state`/`get_state`, values arbitrary binary data [P4], which
could hold rheon's bytes; whether a state update is atomic with the
acknowledgement is not stated [P4]: not verified.

**(c) Arrow or Python path.** A Python consumer through `pulsar-client`
(wheels for CPython 3.10 to 3.14, every platform rheon ships on) [P5], or a
Python Pulsar Function, which handles one message per call [P6]. No Arrow
path; a batch is the consumer's own accumulation. Window functions, which
batch, are Java only [P7].

**(d) Replay and exactly-once.** Broker-side deduplication by producer
sequence ID [P1]; transactions for consume-process-produce [P3]. Functions
offer at-most-once, at-least-once (default) and effectively-once, the last
needing deduplication and a Failover subscription [P7]; exactly-once in
Functions covers only an idempotent function of a single event [P3].

**(e) Static linking.** None in rheon. `pulsar-client` ships per-platform
binary wheels of 4 to 11 MB [P5]; that they wrap the C++ client statically
was not verified.

**(f) Adoption.** An Apache project; its case-studies page lists 47
organizations, among them Intuit, Cisco, Tencent, Discord, Flipkart and
Splunk [P8].

Sources:
[P1] https://pulsar.apache.org/docs/next/concepts-messaging/
[P2] https://pulsar.apache.org/docs/next/concepts-architecture-overview/
[P3] https://pulsar.apache.org/docs/next/txn-why/
[P4] https://pulsar.apache.org/docs/next/functions-develop-state/
[P5] https://pypi.org/project/pulsar-client/
[P6] https://pulsar.apache.org/docs/next/functions-overview/
[P7] https://pulsar.apache.org/docs/next/functions-concepts/
[P8] https://pulsar.apache.org/case-studies/

## 9. NATS JetStream

**(a) Keying and ordering.** JetStream has subjects, not partitions.
Deterministic partitioning maps subject tokens (the group key, say) to a
fixed number of partitions by hash [N1]; without it, durable-consumer
distribution is not partition-aware and messages of one subject can reach
different subscribers [N1]. A redelivery "doesn't slot back into stream
order"; strict order needs `MaxAckPending` of 1, at a cost in throughput
[N2]. So rheon's per-group order holds only with one consumer per partition
and one message in flight, or with rheon restoring order from the stream
sequence number within a batch.

**(b) State and checkpoints.** A durable consumer's position is kept on the
server and advances as messages are acked [N3]. JetStream also has a
replicated key-value store and an object store for larger payloads [N4],
either of which could hold rheon's bytes. Nothing documented makes a write to
them atomic with an ack: not verified, and probably not, so the
state-then-ack order of task 530 would carry over.

**(c) Arrow or Python path.** `nats-py`, a pure-Python client [N5]; pull
consumers fetch in batches, a larger batch meaning fewer round trips [N3]. No
Arrow path.

**(d) Replay and exactly-once.** At-least-once with redelivery until acked;
a consumer can start from the beginning, the latest message, a sequence
number or a time [N4, N2]. Publish deduplication by `Nats-Msg-Id` within a
duplicate window (2 minutes by default) and a double ack (`ack_sync` in
Python) that waits for the server, which the documentation presents as
exactly-once [N6].

**(e) Static linking.** None: the client is pure Python [N5].

**(f) Adoption.** A CNCF project; the server has about 21.1k stars and more
than 40 client implementations [N7].

Sources:
[N1] https://docs.nats.io/nats-concepts/subject_mapping
[N2] https://docs.nats.io/learn/jetstream/delivery-and-acknowledgment
[N3] https://docs.nats.io/nats-concepts/jetstream/consumers
[N4] https://docs.nats.io/nats-concepts/jetstream
[N5] https://pypi.org/project/nats-py/
[N6] https://docs.nats.io/using-nats/developer/develop_jetstream/model_deep_dive
[N7] https://github.com/nats-io/nats-server

## 10. Amazon Kinesis Data Streams

**(a) Keying and ordering.** A partition key is hashed by MD5 to a shard;
each shard is read by one stream-consumer instance at a time, "which
preserves the order of the records in that shard", and each record carries
its sequence number [KN1]. On a reshard, the client library finishes the
parent shard before its children, so records stay in put order [KN2]. A
stream keyed by group gives rheon both needs, like Kafka.

**(b) State and checkpoints.** The Kinesis Client Library (KCL) keeps one
lease per shard and a checkpoint sequence number per shard in a DynamoDB
table [KN2]. Rheon's state would be saved before the checkpoint is written,
as in task 530; the two are separate writes (DynamoDB and wherever the state
goes), not atomic, and `skip_learned` covers the gap.

**(c) Arrow or Python path.** The KCL is Java; Python runs through the
MultiLangDaemon, which needs Java installed [KN2]; `amazon-kclpy` 3.2.0 is
its Python interface [KN3]. Records are opaque blobs [KN1], handed to a
record processor in batches [KN4]. No Arrow path.

**(d) Replay and exactly-once.** At-least-once: producer retries and record
processor restarts both duplicate records, and a moved shard processor
restarts from its last checkpoint [KN4]. AWS's advice is idempotent output
keyed by shard and sequence number [KN4], which is the shape of rheon's
state-then-checkpoint plus `skip_learned`.

**(e) Static linking.** None in rheon; the cost is a JVM beside the Python
consumer for the KCL [KN2].

**(f) Adoption.** A managed AWS service; no public adoption figures were
read: not verified.

Sources:
[KN1] https://docs.aws.amazon.com/streams/latest/dev/key-concepts.html
[KN2] https://docs.aws.amazon.com/streams/latest/dev/shared-throughput-kcl-consumers.html
[KN3] https://pypi.org/project/amazon-kclpy/
[KN4] https://docs.aws.amazon.com/streams/latest/dev/kinesis-record-processor-duplicates.html

## 11. Google Cloud Pub/Sub

**(a) Keying and ordering.** No partitions. Messages with the same ordering
key arrive in publish order once ordering is enabled on the subscription; a
key usually stays with one streaming-pull client while it has outstanding
messages, but affinity can move, so "applications should handle any client
processing a given key" [G1]. A redelivery of one message redelivers every
later message of its key, acknowledged ones too [G1]. Publishing is limited
to 1 MBps per ordering key [G1]. For rheon, a group can move between
consumers at any time, so each consumer would need any group's state on
demand: the bank has no per-group load today.

**(b) State and checkpoints.** No consumer state beyond the subscription's
ack state. Snapshots record that ack state at a point in time and last at
most seven days [G2]. Rheon's state would live outside, saved before the ack;
not atomic.

**(c) Arrow or Python path.** `google-cloud-pubsub` 2.44.0, Python 3.10 or
newer [G3]; messages arrive one by one to a callback (batching by the
consumer's own code). No Arrow path.

**(d) Replay and exactly-once.** Seek to a snapshot, or to a time if
retention of acknowledged messages is configured [G2]. Exactly-once delivery
(pull subscriptions only, in one region) means no redelivery after a
successful ack, not exactly-once processing; publisher retries can still
duplicate, and with ordering, acks must arrive in order, which limits
throughput to thousands of messages a second [G4].

**(e) Static linking.** None in rheon; whether the client depends on
`grpcio` was not stated on the page read: not verified.

**(f) Adoption.** A managed Google Cloud service; no adoption figures read:
not verified.

Sources:
[G1] https://docs.cloud.google.com/pubsub/docs/ordering
[G2] https://docs.cloud.google.com/pubsub/docs/replay-overview
[G3] https://pypi.org/project/google-cloud-pubsub/
[G4] https://docs.cloud.google.com/pubsub/docs/exactly-once-delivery

## 12. RisingWave

**(a) Keying and ordering.** A streaming SQL database; the pages read say
nothing of the order in which an aggregate sees its rows [W1]: not verified,
and SQL gives none without an `ORDER BY`, which the aggregate page does not
mention [W1].

**(b) State and checkpoints.** Checkpoints are Chandy-Lamport snapshots,
every second by default, persisted to remote storage [W2], with barriers
from the meta node [W3]. Internal state updates are exactly-once "without
relying on Kafka transactions" [W4]. The only stateful user code is a
user-defined aggregate (`create_state`, `accumulate`, optional `retract` and
`finish`) [W1], whose state would be checkpointed with the rest; how it is
serialized is not stated [W1].

**(c) Arrow or Python path.** Two kinds of Python function. Embedded ones,
including every aggregate, run in an interpreter inside RisingWave, are
disabled by default, and may import only `json`, `decimal`, `re`, `math`
and `datetime` [W5]: rheon cannot be imported there. External ones run in a
separate UDF server (the `arrow-udf` SDK) reached over gRPC or HTTP/2, with a
`batch=True` option, but are scalar or table functions only, without state,
and one server is held to a core by the GIL [W6]. So rheon could run only as
a stateless external function, keeping its own state outside RisingWave's
checkpoints, which gives up (b).

**(d) Replay and exactly-once.** Recovery restarts from the last checkpoint
[W2]; sinks are connector-dependent, many at-least-once [W4], and the Kafka
sink is at-least-once [W7].

**(e) Static linking.** None in rheon for an external function server.

**(f) Adoption.** About 9.4k stars, Apache-2.0, written in Rust; the README
names no adopters [W8].

Sources:
[W1] https://docs.risingwave.com/sql/commands/sql-create-aggregate
[W2] https://docs.risingwave.com/reference/fault-tolerance
[W3] https://docs.risingwave.com/get-started/architecture
[W4] https://docs.risingwave.com/reference/risingwave-kafka-streams-comparison
[W5] https://docs.risingwave.com/sql/udfs/embedded-python-udfs
[W6] https://docs.risingwave.com/sql/udfs/use-udfs-in-python
[W7] https://docs.risingwave.com/integrations/destinations/apache-kafka (from a search summary; the page itself was not read)
[W8] https://github.com/risingwavelabs/risingwave

## Notes for the later parts

- Every host here keeps state per key, and a rheon bank is one value for all
  its groups. Either rheon gains a per-group save and load, or a host keys by
  shard (a Kafka partition) and holds a whole bank per key, saving it only at
  the host's checkpoint. Which hosts let a function save only at checkpoint
  time is the open question for the ranking.
- No host among the six passes Arrow batches to a stateful Python function:
  PyFlink's Arrow functions are stateless or per window, Spark passes pandas
  frames to its stateful calls, and the Python libraries pass Python objects.
