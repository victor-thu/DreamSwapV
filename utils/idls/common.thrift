include "base.thrift"

namespace go  idl.labcv.gateway.common
namespace py  idl.labcv.gateway.common
namespace cpp idl.labcv.gateway.common

// 用户鉴权参数
struct AuthInfo {
  1: string app_key,
  2: string timestamp,
  3: string nonce,
  4: string sign, // 使用app_secret+nonce+timestamp生成的签名,app_secret为应用秘钥,从智创控制台获取
}

// 同步接口请求结构体
struct AlgoReq {
  1: string req_key,
  2: list<binary> binary_data,
  3: string req_json,
  4: optional binary req_custom_structure,
  5: AuthInfo auth_info,
  255: optional base.Base Base,
}

// 同步接口返回结构体
struct AlgoResp {
  1: list<binary> binary_data,
  2: string resp_json,
  3: optional binary resp_custom_structure,
  255: optional base.BaseResp BaseResp,
}

// 异步回调类型枚举值
enum CallbackType {
  RPC = 0 // need psm, idc, cluster
  HTTP = 1 // need http_url
  EVENTBUS = 2
}

// 异步回调参数
struct TaskCallbackParam {
  1: CallbackType callback_type,
  2: optional string http_url,
  3: optional string psm,
  4: optional string idc,
  5: optional string cluster,
  6: optional string event
}

// 异步任务提交接口请求结构体
struct SubmitTaskReq {
  1: string req_key,
  2: list<binary> binary_data,
  3: string req_json,
  4: optional TaskCallbackParam task_callback_param,
  5: AuthInfo auth_info,
  6: string callback_args,
  7: optional i64 expired_duration, // 过期时间，单位是秒。可以不设置，若设置需要大于0，否则返回报错
  8: optional string template_id, // 视频审核抽帧需要使用，若需要进行视频审核则需要传入
  255: optional base.Base Base,
}

// 异步任务提交接口返回结构体
struct SubmitTaskResp {
  1: string task_id,
  2: string resp_json,
  255: optional base.BaseResp BaseResp,
}

// 异步任务查询接口请求结构体
struct GetResultReq {
  1: string req_key,
  2: string task_id,
  3: string req_json,
  255: optional base.Base Base,
}

// 异步任务查询接口返回结构体
struct GetResultResp {
  1: list<binary> binary_data,
  2: string resp_json,
  3: string status,
  4: string callback_args,
  5: i64 progress, // 任务进度（0-100）
  6: string req_json, // 任务提交时的req_json
  255: optional base.BaseResp BaseResp,
}

// 异步任务状态更新接口请求结构体
struct UpdateTaskReq {
  1: string task_id,
  2: string req_json,
  255: optional base.Base Base,
}

// 异步任务状态更新接口返回结构体
struct UpdateTaskResp {
  1: string resp_json,
  255: optional base.BaseResp BaseResp,
}

// 业务回调接口请求结构体
struct TaskCallbackReq {
  1: string req_key,
  2: string task_id,
  3: string resp_json,
  4: string status,
  5: i32 status_code,
  6: string status_message,
  7: string callback_args,
  8: list<binary> binary_data,
  255: optional base.Base Base,
}

// 业务回调接口返回结构体
struct TaskCallbackResp {
  255: optional base.BaseResp BaseResp, // statusCode = 0为成功，其余为失败
}

// 异步任务取消请求结构体
struct CancelTaskReq {
  1: string req_key,
  2: string task_id,
  3: AuthInfo auth_info,
  255: optional base.Base Base,
}

// 异步任务取消返回结构体
struct CancelTaskResp {
  1: string req_key,
  2: string task_id,
  255: optional base.BaseResp BaseResp,
}
