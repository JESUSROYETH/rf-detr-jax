# Security

This pre-release project has no long-term support or response-time commitment.
Use a pinned, tested environment for deployment and review dependency updates.

## Checkpoints and inputs

The public converter uses PyTorch's restricted weights-only loader by default.
It requires PyTorch 2.10 or newer because earlier restricted loaders have a
[known vulnerability](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
Only use `--trust-checkpoint`, its explicit unsafe legacy option for a checkpoint whose producer and
contents you trust: Python pickle can execute code while loading. Conversion to
NPZ does not establish that a model is correct or that its weights may be redistributed.
Reference/benchmark scripts are developer tooling for trusted inputs; some load
legacy upstream objects and are not an untrusted checkpoint service.

NumPy parameter files are loaded without pickle. Large or malicious archives,
images and annotations can still exhaust resources. Process external datasets
under appropriate resource limits. Dataset paths and compilation-cache locations
must be controlled by the operator; never share a writable executable cache with
untrusted users.

Resumable training checkpoints retain configuration, including dataset/output
locations, to restore the run. Review them before sharing. `export_for_roboflow`
writes a separate local inference bundle with those known training paths removed;
it does not upload anything. Weights, datasets and training checkpoints are not
part of this repository's source distributions.

## Reporting

Use GitHub's private vulnerability reporting feature if it is enabled for the
repository. If unavailable, contact the repository owner through an existing
private channel to arrange disclosure. Do not put credentials, exploit payloads
or private datasets in a public issue. Include the affected revision, a minimal
reproduction, and the expected impact without sensitive data.

Before a public release, maintainers should verify that a private reporting route
is available. Repository scans are heuristic checks, not proof that no secret or
private content exists in the tree or its history.
