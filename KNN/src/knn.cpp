#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <stdint.h>
#include "data_handler.hpp"
#include <algorithm>

knn::knn(int val)
{
    k = val;
}

knn::knn()
{
    //Nothing
}

knn::~knn()
{
    // LoL
}

/*
void knn::find_k_nearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    size_t n = training_data->size();

    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(n);

    // Tüm distance’ları hesapla
    for (size_t j = 0; j < n; ++j)
    {
        double dist = calculate_distance(query_point, training_data->at(j));
        distances.push_back(std::make_pair(dist, j));
    }

    // K’yı aşmamak için limit belirle
    size_t limit = std::min((size_t)k, distances.size());

    // nth_element ile en küçük k elemanı öne getir
    std::nth_element(
        distances.begin(),
        distances.begin() + limit,
        distances.end(),
        [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
            return a.first < b.first;
        });

    // İlk k tanesini neighbors’a ekle
    for (size_t i = 0; i < limit; ++i)
    {
        neighbors->push_back(training_data->at(distances[i].second));
    }
}
*/

void knn::set_training_data(std::vector<data *> *vect)
{
    training_data = vect;
}

void knn::set_test_data(std::vector<data *> *vect)
{
    test_data = vect;
}

void knn::set_validation_data(std::vector<data *> *vect)
{
    validation_data = vect;
}

void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> class_freq;
    for (int i = 0; i < neighbors->size(); i++)
    {
        if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
        {
            class_freq[neighbors->at(i)->get_label()] = 1;
        }
        else
        {
            class_freq[neighbors->at(i)->get_label()]++;
        }
        
    }
    
    int best = 0;
    int max = 0;

    std::map<uint8_t, int>::iterator it;

    for (it = class_freq.begin(); it != class_freq.end(); it++)
    {
        if (it->second > max)
        {
            max = it->second;
            best = it->first;
        }
        
    }
    delete neighbors;
    return best;
}

double knn::calculate_distance(data *query_point, data* input)
{
    const std::vector<uint8_t>& a = *query_point->get_feature_vector();
    const std::vector<uint8_t>& b = *input->get_feature_vector();
    const size_t size = a.size();

    if (size != b.size())
    {
        fprintf(stderr, "Error: Vector size mismatch (%zu vs %zu)\n", a.size(), b.size());
        return std::numeric_limits<double>::infinity();
    }

    double distance = 0.0;

#ifdef EUCLID
    for (size_t i = 0; i < size; ++i)
    {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        distance += diff * diff;
    }
    distance = std::sqrt(distance);

#elif defined MANHATTAN
    for (size_t i = 0; i < size; ++i)
    {
        distance += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
#endif

    return distance;
}


double knn::validate_performance()
{
    double current_performance = 0;
    int count = 0;
    int data_index = 0;
    for (int i = 0; i < validation_data->size(); i++)
    {
        data *query_point = validation_data->at(i);
        find_k_nearest(query_point);
        int prediction = predict();
        //printf("%d -> %d \n", prediction, query_point->get_label());
        if (prediction == query_point->get_label())
        {
            count++;
        }
        
        data_index++;
        printf("Current Performance = %.3f %%\n",((double)count*100.0) /((double)data_index));
    }
    current_performance = ((double)count*100.0) /((double)validation_data->size());
    printf("Validation Performance for K = %d => %.3f %%\n", k , current_performance);
    return current_performance;

}

double knn::test_performance()
{
    double current_performance = 0;
    int count = 0;
    for (int i = 0; i < test_data->size(); i++)
    {
        data * query_point = test_data->at(i);
        find_k_nearest(query_point);
        int prediction = predict();
        if(prediction == query_point->get_label())
        {
            count++;
        }

    }
    
    current_performance = ((double)count*100.0) / ((double)test_data->size());
    printf("Tested Performance = %.3f %%\n", current_performance);
    return current_performance;
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../data/train-images-idx3-ubyte");
    dh->read_feature_labels("../data/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    knn *knearest = new knn();
    knearest->set_training_data(dh->get_training_data());
    knearest->set_test_data(dh->get_test_data());
    knearest->set_validation_data(dh->get_validation_data());
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for (int i = 1; i <= 4; i++)
    {
        if (i == 1)
        {
            knearest->set_k(i);
            performance = knearest->validate_performance();
            best_performance = performance;
        }
        else
        {
            knearest->set_k(i);
            performance = knearest->validate_performance();
            if (performance > best_performance)
            {
                best_performance = performance;
                best_k = i;
            }
            
        }
        
    }
    knearest->set_k(best_k);
    knearest->test_performance();
}