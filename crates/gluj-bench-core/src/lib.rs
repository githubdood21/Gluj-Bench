mod cancellation;
mod model;
mod protocol;
mod provider;
mod registry;

pub use cancellation::CancellationToken;
pub use model::*;
pub use protocol::{
    ProtocolCommand, ProtocolRequest, benchmark_error_response, error_response, parse_request,
    progress_response, result_response, success_response,
};
pub use provider::{BenchmarkProvider, ProgressCallback};
pub use registry::BenchmarkRegistry;

pub const PROTOCOL_VERSION: u32 = 2;
