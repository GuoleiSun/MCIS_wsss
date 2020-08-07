#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/weak_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
WeakSegDataLayer<Dtype>::~WeakSegDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void WeakSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string img_name;
  string gt_name;
  string sp_name;
  while (infile >> img_name >> gt_name >> sp_name) {
    vector<string> line;
    line.push_back(img_name);
    line.push_back(gt_name);
    line.push_back(sp_name);
    lines_.push_back(line);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];

  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  //const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_width = 0;
  int crop_height = 0;
  CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
	|| (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (transform_param.has_crop_size()) {
    crop_width = transform_param.crop_size();
    crop_height = transform_param.crop_size();
  } 
  if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
    crop_width = transform_param.crop_width();
    crop_height = transform_param.crop_height();
  }

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_width > 0 && crop_height > 0) {
    top[0]->Reshape(batch_size, channels, crop_height, crop_width);
    this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, crop_height, crop_width);
    }

    //label
    top[1]->Reshape(batch_size, 1, crop_height, crop_width);
    this->transformed_label_.Reshape(batch_size, 1, crop_height, crop_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, crop_height, crop_width);
    }

    //sp_mask
    top[2]->Reshape(batch_size, 1, crop_height, crop_width);
    this->transformed_sp_mask_.Reshape(batch_size, 1, crop_height, crop_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].sp_mask_.Reshape(batch_size, 1, crop_height, crop_width);
    }
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
    }

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    this->transformed_label_.Reshape(batch_size, 1, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
    }

    //sp_mask
    top[2]->Reshape(batch_size, 1, height, width);
    this->transformed_sp_mask_.Reshape(batch_size, 1, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].sp_mask_.Reshape(batch_size, 1, height, width);
    }
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  LOG(INFO) << "output sp_mask size: " << top[2]->num() << ","
	    << top[2]->channels() << "," << top[2]->height() << ","
	    << top[2]->width();
}

template <typename Dtype>
void WeakSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void WeakSegDataLayer<Dtype>::load_batch(WeakSegBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(batch->label_.count());
  CHECK(batch->sp_mask_.count());
  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_label_.count());
  CHECK(this->transformed_sp_mask_.count());

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  Dtype* prefetch_data     = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label    = batch->label_.mutable_cpu_data(); 
  Dtype* prefetch_sp_mask = batch->sp_mask_.mutable_cpu_data();

  const int lines_size = lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    vector<cv::Mat> mats;

    int img_row, img_col;
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0], new_height, new_width, is_color, &img_row, &img_col);
    CHECK(cv_img.data) << "Fail to load img: " << root_folder + lines_[lines_id_][0];
    cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_][1], new_height, new_width, false);
    CHECK(cv_gt.data) << "Fail to load seg: " << root_folder + lines_[lines_id_][1];
    cv::Mat cv_sp = ReadMaskToCVMat(root_folder + lines_[lines_id_][2], new_height, new_width, true);
    CHECK(cv_sp.data) << "Fail to load sp_mask: " << root_folder + lines_[lines_id_][2];

    const int height = cv_img.rows;
    const int width = cv_img.cols;
    const int gt_channels = cv_gt.channels();
    const int gt_height = cv_gt.rows;
    const int gt_width = cv_gt.cols;
    const int sp_channels = cv_sp.channels();
    const int sp_height = cv_sp.rows;
    const int sp_width = cv_sp.cols;
    CHECK_EQ(height, gt_height);
    CHECK_EQ(width, gt_width);
    CHECK_EQ(gt_channels, 1);
    CHECK_EQ(height, sp_height);
    CHECK_EQ(width, sp_width);
    CHECK_EQ(sp_channels, 1);
    mats.push_back(cv_img);
    mats.push_back(cv_gt);
    mats.push_back(cv_sp);

    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply transformations (mirror, crop...) to the image
    int offset;
    offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(prefetch_label + offset);

    offset = batch->sp_mask_.offset(item_id);
    this->transformed_sp_mask_.set_cpu_data(prefetch_sp_mask + offset);

    this->data_transformer_->TransformWeakSeg(mats, &(this->transformed_data_), 
            &(this->transformed_label_), &(this->transformed_sp_mask_), ignore_label);
    
    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	    ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(WeakSegDataLayer);
REGISTER_LAYER_CLASS(WeakSegData);

}  // namespace caffe
