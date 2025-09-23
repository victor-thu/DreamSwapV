include "./base.thrift"

namespace py ecom.goods_quality.image_quality
namespace go ecom.goods_quality.image_quality

struct ImageQualityReq {
    1: i64 product_id,
    2: string image_url,
    3: i64 first_category_id,
    4: optional bool is_s_product,
    5: optional binary image_data,
    6: optional i64 second_category_id,
    7: optional i64 third_category_id,
    255: optional base.Base Base
}

struct ImageQualityRes {
    1: i32 tag, //0是非优质画风，1是优质画风
    2: i32 subject_incomplete_tag, //1表示命中无主体or主体不完整，0表示未命中
    3: i32 background_clear_tag, //1表示命中低质图or展示不清晰or非整洁背景，0表示未命中
    4: i32 fake_model_tag, //1表示命中假模，0表示未命中
    5: i32 splice_tag, //1表示命中拼接，0表示未命中
    6: i32 psoriasis_tag, //1表示命中牛皮癣，0表示未命中
    7: i32 border_tag, //1表示命中白边or黑白，0表示未命中
    8: i32 language_tag, //1表示命中不可接受的语言，0表示未命中
    9: i32 unknown_tag, //1表示命中其他未知理由，0表示未命中
    255: optional base.BaseResp BaseResp
}

service ImageQualityService {
  ImageQualityRes Detect(1: ImageQualityReq req)
}