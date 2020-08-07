#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/channel_wise_scale_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ChannelWiseScaleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ChannelWiseScaleLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 6, 3, 4)),
        blob_bottom_scale_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    vector<int> scale_shape;
    scale_shape.push_back(6);
    scale_shape.push_back(1);
    scale_shape.push_back(1);
    scale_shape.push_back(1);
    blob_bottom_scale_->Reshape(scale_shape);
    FillerParameter filler_param;
    filler_param.set_std(0.1);
    GaussianFiller<Dtype> filler(filler_param);
    //filler_param.set_max(1);
    //filler_param.set_min(1);
    //UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_scale_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ChannelWiseScaleLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_scale_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_scale_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ChannelWiseScaleLayerTest, TestDtypesAndDevices);
/*
TYPED_TEST(ChannelWiseScaleLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  layer_param.mutable_channel_wise_scale_param()->set_scale_dim(3);
  shared_ptr<ChannelWiseScaleLayer<Dtype> > layer(new ChannelWiseScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const int outer_num = 3;
  const int inner_num = 12;
  const int class_num = 2;
  const Dtype* scale_data = this->blob_bottom_scale_->cpu_data();
  const int count = this->blob_bottom_->count() / 9;
  const int dim = this->blob_bottom_->count() / outer_num;
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num; ++j) {
      for (int c = 0; c < class_num; ++c) {
        Dtype value = 0;
        for (int s = 0; s < 3; ++s) {
          value += in_data[i * dim + (s * class_num + c) * inner_num + j] * scale_data[s * class_num + c];
        }
        EXPECT_NEAR(data[i * count + c * inner_num + j], value, 1e-5);
      }
    }
  }
}
*/

TYPED_TEST(ChannelWiseScaleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChannelWiseScaleParameter *scale_param = layer_param.mutable_channel_wise_scale_param();
  scale_param->set_scale_dim(3);
  scale_param->mutable_weight_filler()->set_type("gaussian");
  scale_param->mutable_weight_filler()->set_std(0.1);
  ChannelWiseScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ChannelWiseScaleLayerTest, TestGradientScale) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  ChannelWiseScaleParameter *scale_param = layer_param.mutable_channel_wise_scale_param();
  scale_param->set_scale_dim(3);
  ChannelWiseScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}


/*
TYPED_TEST(ChannelWiseScaleLayerTest, TestGradientScaleAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_channel_wise_scale_param()->set_axis(2);
  layer_param.mutable_channel_wise_scale_param()->set_scale_dim(3);
  ChannelWiseScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}
*/
}  // namespace caffe
