#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/hybrid_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class HybridCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  HybridCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_soft_label_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_hard_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_soft_label_);
    for (int i = 0; i < blob_bottom_soft_label_->count(); ++i) {
      blob_bottom_soft_label_->mutable_cpu_data()[i] = std::abs(blob_bottom_soft_label_->mutable_cpu_data()[i]);
    }
    Dtype* prob_data = blob_bottom_soft_label_->mutable_cpu_data();
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 6; ++j) {
        Dtype sum = 0;
        for (int c = 0; c < 5; ++c) {
          if (c == 0 || c == 3) {
            prob_data[i * 30 + c * 6 + j] = 0;
          }
        }
        for (int c = 0; c < 5; ++c) {
          sum += prob_data[i * 30 + c * 6 + j];
        }
        for (int c = 0; c < 5; ++c) {
          prob_data[i * 30 + c * 6 + j] /= sum;
        }
      }
      if (i % 3 == 0) {
        for (int j = 5; j < 6; ++j) {
          for (int c = 0; c < 5; ++c)
            prob_data[i * 30 + c * 6 + j] = -1;
        }
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_soft_label_);
    const Dtype* soft_label = blob_bottom_soft_label_->cpu_data();
    Dtype* hard_label = blob_bottom_hard_label_->mutable_cpu_data();
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 6; ++j) {
        Dtype label_value = soft_label[i * 30 + j];
        if (label_value < 0) {
          hard_label[i * 6 + j] = -1;
        } else {
          Dtype max_prob = 0;
          Dtype max_pos = 0;
          for (int c = 0; c < 5; ++c) {
            if (max_prob > soft_label[i * 30 + c * 6 + j]) {
              max_prob = soft_label[i * 30 + c * 6  + j];
              max_pos = c;
            }
          }
          hard_label[i * 6 + j] = max_pos; 
        }
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_hard_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HybridCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_soft_label_;
    delete blob_bottom_hard_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_soft_label_;
  Blob<Dtype>* const blob_bottom_hard_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HybridCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(HybridCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_loss_param()->set_hybrid_thresh(0.5);
  HybridCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HybridCrossEntropyLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  layer_param.mutable_loss_param()->set_hybrid_thresh(0.5);
  HybridCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
