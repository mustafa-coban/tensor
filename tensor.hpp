
#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cassert>


template<class T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<Arithmetic ComponentType>
class Tensor {
public:
    // Constructs a tensor with rank = 0 and zero-initializes the element.
    Tensor();

    // Constructs a tensor with arbitrary shape and zero-initializes all elements.
    Tensor(const std::vector<size_t> &shape);

    // Constructs a tensor with arbitrary shape and fills it with the specified value.
    explicit Tensor(const std::vector<size_t> &shape, const ComponentType &fillValue);

    // Constructs a tensor with arbitrary shape and data.
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<ComponentType> &data);

    // Copy-constructor.
    Tensor(const Tensor<ComponentType> &other);

    // Move-constructor.
    Tensor(Tensor<ComponentType> &&other) noexcept;

    // Copy-assignment
    Tensor &
    operator=(const Tensor<ComponentType> &other);

    // Move-assignment
    Tensor &
    operator=(Tensor<ComponentType> &&other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Returns the rank of the tensor.
    [[nodiscard]] size_t rank() const;

    // Returns the shape of the tensor.
    [[nodiscard]] std::vector<size_t> shape() const;

    // Returns the number of elements of this tensor.
    [[nodiscard]] size_t numElements() const;

    // Element access function
    const ComponentType &
    operator()(const std::vector<size_t> &idx) const;

    // Element mutation function
    ComponentType &
    operator()(const std::vector<size_t> &idx);

    bool checkEqual(const Tensor<ComponentType> &other) const;

    // Element access function
    const ComponentType &
    operator[](const size_t &idx) const;

    // Element mutation function
    ComponentType &
    operator[](const size_t &idx);

private:
    std::vector<size_t> _shape;
    std::vector<ComponentType> _data;
    std::vector<size_t> _indexing;

    [[nodiscard]] size_t _calculateIndex(const std::vector<size_t> &idx) const;

    void _fillIndexing();
};

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const std::vector<size_t> &shape, const std::vector<ComponentType> &data): _shape(
        shape.begin(), shape.end()),
                                                                                                         _data(data),
                                                                                                         _indexing(
                                                                                                                 shape) {

    _data = data;
    this->_fillIndexing();
}

template<Arithmetic ComponentType>
bool Tensor<ComponentType>::checkEqual(const Tensor<ComponentType> &other) const {
    if (_shape != other._shape) {
        return false;
    }
    if (_data != other._data) {
        return false;
    }
    return true;
}


template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor() : _shape(1, 0), _data(1, 0), _indexing(1, 0) {

}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const std::vector<size_t> &shape): _shape(shape), _data(this->numElements(), 0),
                                                                 _indexing(shape) {
    if (_shape.empty()) {
        _shape.push_back(0);
        _data.push_back(0);
        _indexing.push_back(0);
        return;
    }
    this->_fillIndexing();
}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const std::vector<size_t> &shape, const ComponentType &fillValue): _shape(shape),
                                                                                                 _data(this->numElements(),
                                                                                                       fillValue),
                                                                                                 _indexing(shape) {
    this->_fillIndexing();
}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const Tensor<ComponentType> &other): _shape(other._shape), _data(other._data),
                                                                   _indexing(other._indexing) {

}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(Tensor<ComponentType> &&other) noexcept {
    _shape = std::move(other._shape);
    _data = std::move(other._data);
    _indexing = std::move(other._indexing);
    other._shape = {0};
    other._data = {0};
    other._indexing = {0};
}

template<Arithmetic ComponentType>
Tensor<ComponentType> &Tensor<ComponentType>::operator=(const Tensor<ComponentType> &other) {
    _shape = other.shape();
    _data = other._data;
    _indexing = other._indexing;
    return *this;
}

template<Arithmetic ComponentType>
Tensor<ComponentType> &Tensor<ComponentType>::operator=(Tensor<ComponentType> &&other) noexcept {
    _shape = std::move(other._shape);
    _data = std::move(other._data);
    _indexing = std::move(other._indexing);
    other._shape = {0};
    other._data = {0};
    other._indexing = {0};
    return *this;
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::rank() const {
    return this->_shape.size();
}

template<Arithmetic ComponentType>
std::vector<size_t> Tensor<ComponentType>::shape() const {
    return this->_shape;
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::numElements() const {
    size_t _numElements = 1;
    for (auto &element: _shape) {
        _numElements *= element;
    }
    return _numElements;
}

template<Arithmetic ComponentType>
const ComponentType &Tensor<ComponentType>::operator()(const std::vector<size_t> &idx) const {
    assert(idx.size() == _shape.size() || (idx.empty() && _shape[0] == 0));
    return _data[_calculateIndex(idx)];
}

template<Arithmetic ComponentType>
ComponentType &Tensor<ComponentType>::operator()(const std::vector<size_t> &idx) {
    assert(idx.size() == _shape.size() || (idx.empty() && _shape[0] == 0));
    return _data[_calculateIndex(idx)];
}

template<Arithmetic ComponentType>
ComponentType &Tensor<ComponentType>::operator[](const size_t &idx) {
    return _data[idx];
}

template<Arithmetic ComponentType>
const ComponentType &Tensor<ComponentType>::operator[](const size_t &idx) const {
    return _data[idx];
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::_calculateIndex(const std::vector<size_t> &idx) const {
    size_t index = 0;
    for (auto currentIndex = idx.begin(), multiplier = _indexing.begin();
         currentIndex != idx.end() && multiplier != _indexing.end(); ++currentIndex, ++multiplier) {
        index += *currentIndex * *multiplier;
    }
    return index;
}

template<Arithmetic ComponentType>
void Tensor<ComponentType>::_fillIndexing() {
    for (auto cur = _indexing.rbegin(), next = _indexing.rbegin() + 1; next != _indexing.rend(); ++cur, ++next) {
        *next *= *cur;
    }
    std::rotate(_indexing.begin(), _indexing.begin() + 1, _indexing.end());
    _indexing.back() = 1;
}

// Returns true if the shapes and all elements of both tensors are equal.
template<Arithmetic ComponentType>
bool operator==(const Tensor<ComponentType> &a, const Tensor<ComponentType> &b) {
    return a.checkEqual(b);
}

// Pretty-prints the tensor to stdout.
// This is not necessary (and not covered by the tests) but nice to have, also for debugging (and for exercise of course...).
template<Arithmetic ComponentType>
std::ostream &
operator<<(std::ostream &out, const Tensor<ComponentType> &tensor) {
    // TODO (optional): Implement some nice stdout printer for debugging/exercise.
    std::vector<size_t> traverse(tensor.rank(), 0);
    std::vector<size_t> shape = tensor.shape();

    if (tensor.rank() == 0) {
        return out;
    }

    if (tensor.rank() == 1) {
        for (size_t i = 0; i < shape.front(); ++i) {
            out << tensor[i] << " ";
        }
        out << std::endl;
        return out;
    }

    if (tensor.rank() == 2) {
        for (size_t row = 0; row < shape.front(); ++row) {
            for (size_t col = 0; col < shape.back(); ++col) {
                out << tensor({row, col}) << " ";
            }
            out << std::endl;
        }
        return out;
    }

    while (traverse.front() < shape.front()) {
        const size_t startIndex = shape.size() - 3;
        std::string deliminator(shape.back() * 2, '=');
        // print matrix of tensor part with info
        // info is indexes before last two
        out << deliminator << std::endl;
        out << "{ ";
        for (size_t i = 0; i <= startIndex; ++i) {
            out << traverse[i] << " ";
        }
        out << "}" << std::endl;
        //matrix
        for (size_t row = 0; row < shape.front(); ++row) {
            for (size_t col = 0; col < shape.back(); ++col) {
                traverse.back() = col;
                traverse[traverse.size()-2] = row;
                out << tensor(traverse) << " ";
            }
            out << std::endl;
        }

        out << deliminator << std::endl;

        for (size_t last = startIndex; last <= startIndex;) {
            traverse[last]++;
            if (traverse[last] != shape[last]) {
                break;
            }

            if (last == 0) {
                break;
            }
            traverse[last] = 0;
            last--;
        }
    }
    return out;
}

// Reads a tensor from file.
template<Arithmetic ComponentType>
Tensor<ComponentType> readTensorFromFile(const std::string &filename) {
    std::ifstream ifs(filename, std::ifstream::in);

    if (!ifs.is_open()) {
        std::cerr << "Error opening file." << filename << std::endl;
        exit(1);
    }

    size_t rank;
    ifs >> rank;
    std::vector<size_t> shape(rank);
    for (auto &element: shape) {
        ifs >> element;
    }

    size_t numElements = 1;
    for (auto &element: shape) {
        numElements *= element;
    }
    std::vector<ComponentType> data(numElements);
    for (auto &element: data) {
        ifs >> element;
    }
    Tensor<ComponentType> tensor(shape, data);
    return tensor;
}

// Writes a tensor to file.
template<Arithmetic ComponentType>
void writeTensorToFile(const Tensor<ComponentType> &tensor, const std::string &filename) {
    std::ofstream ofs(filename, std::ofstream::out);

    if (!ofs.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    ofs << tensor.rank() << std::endl;

    std::vector<size_t> shape = tensor.shape();

    for (auto &element: shape) {
        ofs << element << std::endl;
    }

    int size = tensor.numElements();
    for (int i = 0; i < size; ++i) {
        ofs << tensor[i] << std::endl;
    }

}
