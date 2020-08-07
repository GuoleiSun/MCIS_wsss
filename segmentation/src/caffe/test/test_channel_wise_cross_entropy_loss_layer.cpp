#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/channel_wise_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class ChannelWiseCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ChannelWiseCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 5, 4, 3)),
        blob_bottom_label_(new Blob<Dtype>(2, 1, 4, 3)),
        blob_bottom_image_label_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    for (int i = 0; i < blob_bottom_image_label_->count(); ++i) {
      blob_bottom_image_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_image_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ChannelWiseCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_image_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_image_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ChannelWiseCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ChannelWiseCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  ChannelWiseCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

//TYPED_TEST(ChannelWiseCrossEntropyLossLayerTest, TestGradientUnnormalized) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  layer_param.mutable_loss_param()->set_normalize(false);
//  ChannelWiseCrossEntropyLossLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_, 0);
//}
}  // namespace caffe
