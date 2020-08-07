#include <algorithm>
#include <cfloat>
#include <vector>
#include <string.h>
#include "caffe/layers/superpixel_pooling_test_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;
// bottom[0] : feature map
// bottom[1] : segment region labels
// bottom[2] : superpixel labels
// top[0]: 1 * 1 * S * 64

template <typename Dtype>
void SuperpixelPoolingTestLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_sp_label = bottom[1]->cpu_data();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //std::cout << "val" << std::endl;
  const Dtype* max_label = std::max_element(bottom_sp_label, bottom_sp_label + bottom[1]->count());
  numsp_ = static_cast<int>(*max_label) + 1;  //number of superpixel

  vector<int> top_shape(4, 1);
  top_shape[0] = numsp_;
  top_shape[1] = channels_;
  max_idx_.Reshape(top_shape);   // use for max pooling

  vector<int> ave_shape(4, 1);
  ave_shape[0] = numsp_;
  ave_size_.Reshape(ave_shape);  // use for average pooling
  
  top[0]->Reshape(top_shape);  // pooling feature
}

template <typename Dtype>
void SuperpixelPoolingTestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

//first part: superpixel pooling
  const Dtype* bottom_data = bottom[0]->cpu_data();   // 1 * 64 * h * w
  const Dtype* bottom_sp_label = bottom[1]->cpu_data();  // 1 * 1 * h * w

  Dtype* top_data = top[0]->mutable_cpu_data();
  int* max_mask = NULL;
  int* ave_mask = NULL;

  switch (this->layer_param_.superpixel_pooling_param().pool_type()) {
    case SuperpixelPoolingTestParameter_PoolType_MAX:
      caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
      max_mask = max_idx_.mutable_cpu_data();
      caffe_set(numsp_*channels_, -1, max_mask);
      for(int i = 0; i < height_; i++) {
        for(int j = 0; j < width_; j++) {
          int label = static_cast<int>(bottom_sp_label[i*width_ + j]);
          if(label == -1) continue;
          // exact max id and value
          for(int c = 0; c < channels_; c++) {
            Dtype bdat = bottom_data[(c*height_ + i) * width_ + j];
            if(top_data[label*channels_ + c] < bdat) {
              top_data[label*channels_ + c] = bdat;
              max_mask[label*channels_ + c] = i*width_ + j;
            }
          }
        }
      }
      break;

    case SuperpixelPoolingTestParameter_PoolType_AVE:
      caffe_set(top[0]->count(), Dtype(0), top_data);
      ave_mask = ave_size_.mutable_cpu_data();
      caffe_set(numsp_, 0, ave_mask);   // per superpixel size
      for(int i = 0; i < height_; i++) {
        for(int j = 0; j < width_; j++) {
          int label = static_cast<int>(bottom_sp_label[i*width_ + j]);
          if(label == -1) continue;
          for(int c = 0; c < channels_; c++) {
            top_data[label*channels_ + c] += bottom_data[(c*height_ + i) * width_ + j];
          }
          ave_mask[label] ++;
        }
      }

      for(int n = 0; n < numsp_; n++) {
        caffe_scal(channels_, Dtype(1) / ave_mask[n], top_data+n*channels_);
      }
      //std::cout << "hahah1" << std::endl;
      break;

    case SuperpixelPoolingTestParameter_PoolType_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void SuperpixelPoolingTestLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_sp_label = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int* max_mask = NULL;
  const int* ave_mask = NULL;

  switch (this->layer_param_.superpixel_pooling_param().pool_type()) {
  case SuperpixelPoolingTestParameter_PoolType_MAX:
    max_mask = max_idx_.cpu_data();
    for(int i = 0; i < numsp_; i++) {
      for(int j = 0; j < channels_; j++) {
        bottom_diff[j*height_*width_ + max_mask[i*channels_ + j]] = top_diff[i*channels_ + j];
      }
    }
    break;
  case SuperpixelPoolingTestParameter_PoolType_AVE:
    ave_mask = ave_size_.cpu_data();
    for(int i = 0; i < height_; i++) {
      for(int j = 0; j < width_; j++) {
        int label = static_cast<int>(bottom_sp_label[i*width_ + j]);
        if(label == -1) continue;
        for(int c = 0; c < channels_; c++) {
          bottom_diff[(c*height_ + i)*width_ + j] = top_diff[label*channels_ + c] / ave_mask[label];
        }
      }
    }
    break;
  case SuperpixelPoolingTestParameter_PoolType_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(SuperpixelPoolingTestLayer);
#endif

INSTANTIATE_CLASS(SuperpixelPoolingTestLayer);
REGISTER_LAYER_CLASS(SuperpixelPoolingTest);
}  // namespace caffe
