include "common.thrift"

namespace go idl.labcv.gateway.vproxy
namespace py idl.labcv.gateway.vproxy
namespace cpp idl.labcv.gateway.vproxy

service VisionService {
  //// 业务方调用
  // 同步接口
  common.AlgoResp Process(1: common.AlgoReq req),

  // 异步任务提交接口
  common.SubmitTaskResp SubmitTask(1: common.SubmitTaskReq req),
  // 异步任务查询接口
  common.GetResultResp GetResult(1: common.GetResultReq req),
  // 异步任务回调接口
  common.TaskCallbackResp DoTaskCallback(1: common.TaskCallbackReq req),
  // 异步任务取消接口
  common.CancelTaskResp CancelTask(1: common.CancelTaskReq req),

  //// 内部服务调用
  // 异步任务状态更新接口
  common.UpdateTaskResp UpdateTask(1: common.UpdateTaskReq req),
}
