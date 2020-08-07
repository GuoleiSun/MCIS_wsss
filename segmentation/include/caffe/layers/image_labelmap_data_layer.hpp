
#ifndef CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_
#define CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class ImageLabelmapDataLayer : public BasePrefetchingLabelmapDataLayer<Dtype> {
 public:
  explicit ImageLabelmapDataLayer(const LayerParameter& param)
      : BasePrefetchingLabelmapDataLayer<Dtype>(param) {}
  virtual ~ImageLabelmapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageLabelmapData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; } //could be three if considering label

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(LabelmapBatch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
  bool normalize_;
};

}

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
