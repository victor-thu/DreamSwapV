include "../base.thrift"

namespace py idls.iccv 
namespace go idls.iccv
namespace java idls.iccv

struct SegmentRequest {
    1: binary image,
    2: list<string> req_type_list,
    255: base.Base Base
}

struct SegmentResponse {
    1: map<string, binary> results,
    255: base.BaseResp BaseResp
}

service SegmentInferenceService {
    SegmentResponse Process(1: SegmentRequest req)
}