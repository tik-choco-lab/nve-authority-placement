# Dataset Design

## 1. Purpose and scope

These synthetic workloads support controlled comparisons of authority-placement
policies under changing interaction demand. They do not reproduce or predict
real MMORPG player behavior.

```text
Configuration + seed -> workload generator -> shared input trace
                                             -> policy replay -> policy outputs
```

Policy decisions never feed back into generation. Every policy can therefore
be evaluated on the same input trace.

## 2. Model

| Component | Representation |
|---|---|
| Peer | Identifier, initial position, mobility model, and sampled positions. |
| Shared entity | Identifier, fixed position, initial authority, and abstract state size. |
| Network | A fixed RTT matrix covering every ordered peer pair, independent of spatial coordinates. |
| Interaction | Timestamp, request ID, entity ID, requester ID, and request type. |
| Authority | One initial holder per entity; subsequent placement is determined by the replayed policy. |

Entity state is represented by `state_size_bytes`, without simulating health,
AI state, or other application details. Interaction weights are stored separately
in `request_types.csv` rather than repeated in the interaction trace.

## 3. Random streams and generation order

Independent pseudorandom streams are derived from the master seed and a namespace.
The generator hashes the UTF-8 string `"<seed>:<namespace>"` with SHA-256 and
uses the first eight bytes as a big-endian integer seed for `random.Random`.
Namespaces separate peer initialization, mobility, entity initialization,
network generation, per-entity arrivals, scenario requester selection, and
per-entity request types.

Changing network or mobility settings therefore does not alter the arrival
sequence. Poisson arrivals use exponentially distributed interarrival times
per entity; fixed-interval arrivals use `1 / rate_per_entity`.
Events from all entities are stably sorted by `(timestamp, entity_id, peer_id)`
before request IDs are assigned.

## 4. Scenario semantics

| Scenario | Requester selection |
|---|---|
| `uniform` | Select each requester uniformly from all peers. |
| `concentrated` | Assign a dominant peer per entity using the seed. Select it with probability `dominant_peer_ratio`; distribute the remaining probability uniformly among other peers. |
| `shifting` | Starting from each entity's initial dominant peer, advance along the peer-ID list every `phase_duration` seconds. Within each phase, use the concentrated distribution. |
| `burst` | Use the concentrated distribution during configured burst intervals and the uniform distribution outside them. |

Dominant-peer assignments and phase/burst information are recorded in the
manifest so that the expected distributions can be checked. They are generation
metadata, not ground-truth authority labels or information for online policies.

## 5. Time and causality

Interaction timestamps are continuous-valued. Mobility advances in
`simulation.timestep` increments, while positions are recorded every
`peers.position_sample_interval`. Separating integration and recording limits
trace size. Static peers have a single position sample at `t=0`.

Online policies may observe requests only up to the current timestamp. Future
windows are available only to offline analysis or the oracle. The generator
does not embed predictions or training/test splits.

## 6. Network and virtual space

Both homogeneous and heterogeneous RTT models draw uniformly from their
configured ranges; the heterogeneous range is wider in the default configuration.
RTTs are symmetric by default, and self RTT is zero. They are independent of
two-dimensional coordinates, so spatial proximity does not imply network proximity.

Random-walk peers choose a direction at each integration step, move at constant
speed, and reflect at world boundaries. Entities remain stationary.

`peers.target_coupling` is optional and disabled by default. Normally, the
dominant-peer assignment does not depend on peer coordinates. When coupling is
enabled, a peer currently dominant for one or more entities directs a step toward
the centroid of those entities with probability `bias_probability`, instead of
choosing a uniform random direction. The speed and boundary rules are unchanged.
This option requires random-walk mobility and an explicit probability in `[0, 1]`.
Omitting or disabling it preserves the uncoupled generation behavior.

In the shifting scenario, a new physical peer becomes dominant at each phase
boundary. Coupling is effective only if the speed and phase duration allow that
peer to approach the entity before the phase ends. The range sweep therefore uses
a smaller world and faster mobility; see `experiments/range_sweep.py` for settings.

## 7. Metrics supported by the inputs

| Metric | Required information |
|---|---|
| Interaction latency | Requester, current authority, pairwise RTT, and processing delay. |
| Migration count, rate, and holding time | Authority transitions produced by the policy. |
| Migration cost | Entity state size, old/new holder RTT, and an optional bandwidth assumption. |
| Network traffic | Request and response sizes plus migration bytes. |
| Mean, median, p95, and maximum latency | Per-request latency measurements. |

Request/response sizes and migration bandwidth are evaluation parameters rather
than facts imposed by the input dataset. Interaction weights are also configurable;
changing their values does not require changing `interactions.csv`.

## 8. Validation and regeneration

Validation checks references, timestamp order and range, unique request IDs,
nonnegative and complete RTT matrices, configured symmetry, zero self RTT,
initial authorities, position samples and bounds, the initial position sample,
request-type consistency, and input checksums.

The distribution tolerance is

```text
absolute_tolerance + sigma * sqrt(p * (1 - p) / n)
```

where `p` is the expected probability and `n` is the relevant sample count.
Uniform workloads are checked per peer; other scenarios are checked for the
eligible dominant-peer share. Both tolerance terms are configurable to accommodate
small samples.

`nve-dataset reproduce-check` regenerates the effective configuration in a
temporary directory and compares SHA-256 hashes for all input CSV files.

## 9. Scaling

Peer count, entity count, duration, and seed lists are configurable.
The dense network matrix requires `O(P^2)` entries. Position traces require
`O(P * duration / position_sample_interval)` samples. Runtime and storage should
be checked for the selected scale; the oracle has an additional quadratic
dependence on peer count per request.

Possible future storage changes include representing the network by its generation
rule and seed, or recording positions only at request times. Such changes would
require a dataset-format version change.

## 10. Explicit assumptions

- Entities are stationary; peers use static or random-walk mobility.
- The interaction rate is specified per entity.
- The default request type is `attack`; additional types can be configured.
- RTTs are fixed over a run and symmetric by default.
- Initial authorities are sampled uniformly using the seed, independently of the evaluated policy.
- Authority selection, interpretation of criticality, migration thresholds, holding times, and prediction algorithms belong to the evaluation layer.
