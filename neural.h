

#if !defined(NEURAL_H)
#define NEURAL_H

#include <functional>
#include <vector>

namespace Neural {
    typedef std::vector<double> Vector;
    typedef std::vector<Vector> Matrix;
    struct Network {
        size_t inputCount;
        size_t hiddenCount;
        size_t outputCount;
        Vector weightsHidden;
        Vector biasesHidden;
        Vector weightsOutput;
        Vector biasesOutput;
        Vector Predict(const Vector& input) const;
        Vector Predict(const Vector& input, Vector& hidden, Vector& output) const;
    };

    struct Trainer {
        Network network;
        Vector hidden;
        Vector output;
        Vector gradHidden;
        Vector gradOutput;
        static Trainer Create(Neural::Network&& network, size_t hiddenCount, size_t outputCount);
        static Trainer Create(size_t inputCount, size_t hiddenCount, size_t outputCount, std::function<double()> rand);
        void Train(const Vector& input, const Vector& output, double lr);
    };
}

#endif
