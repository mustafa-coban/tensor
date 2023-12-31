#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cassert>
#include <numeric>


template<class T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Iterator for Tensor class
template<Arithmetic ComponentType, typename IteratorType>
class Iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ComponentType;
    using difference_type = std::ptrdiff_t;
    using pointer = ComponentType *;
    using reference = typename std::iterator_traits<IteratorType>::reference;

    Iterator(IteratorType it, int alignment) : _it(it), _alignment(alignment) {}

    Iterator &operator++() {
        _it += _alignment;
        return *this;
    }

    Iterator operator++(int) {
        _it += _alignment;
        return *this;
    }

    Iterator operator+(const int advance) {
        _it += (_alignment * advance);
        return *this;
    }

    Iterator operator-(const int advance) {
        _it -= (_alignment * advance);
        return *this;
    }

    bool operator==(const Iterator &other) const {
        return _it == other._it;
    }

    bool operator!=(const Iterator &other) const {
        return _it != other._it;
    }


    reference &operator*() {
        return *_it;
    }


private:
    IteratorType _it;
    int _alignment = 1;
};


template<Arithmetic ComponentType>
class Tensor {
public:


    // Constructs a tensor with rank = 0 and zero-initializes the element.
    Tensor();

    // Constructs a tensor with arbitrary shape and zero-initializes all elements.
    explicit Tensor(const std::vector<size_t> &shape);

    // Constructs a tensor with arbitrary shape and fills it with the specified value.
    explicit Tensor(const std::vector<size_t> &shape, const ComponentType &fillValue);

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

    static Tensor<ComponentType> readFromFile(const std::string &filename);

    void writeToFile(const std::string &filename) const;

    // Reshape the tensor
    // ** Invalidates all iterator acquired before ** undefined behaviour
    void reshape(const std::vector<size_t> &newShape);

    // Iterator-related member functions
    using Iterator_type = Iterator<ComponentType, typename std::vector<ComponentType>::iterator>;
    using Const_Iterator_type = Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>;

    // Returns Iterator for begin
    [[nodiscard]] Iterator_type begin(const std::vector<int> &idx);

    // Returns Iterator for end
    [[nodiscard]] Iterator_type end(const std::vector<int> &idx);

    // Returns Iterator for begin
    [[nodiscard]] Const_Iterator_type begin(const std::vector<int> &idx) const;

    // Returns Iterator for end
    [[nodiscard]] Const_Iterator_type end(const std::vector<int> &idx) const;

    // Returns underlying vector
    [[nodiscard]] std::vector<ComponentType> &data();

    // Returns underlying vector
    [[nodiscard]] const std::vector<ComponentType> &data() const;

private:
    std::vector<size_t> _shape;
    std::vector<ComponentType> _data;

private:
    std::vector<size_t> _indexing;

    [[nodiscard]] size_t _calculateIndex(const std::vector<size_t> &idx) const;

    [[nodiscard]] size_t _calculateIndex(const std::vector<int> &idx) const;

    void _fillIndexing();
};


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
    if (this == &other) return *this;
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
inline size_t Tensor<ComponentType>::rank() const {
    return this->_shape.size();
}

template<Arithmetic ComponentType>
inline std::vector<size_t> Tensor<ComponentType>::shape() const {
    return this->_shape;
}

template<Arithmetic ComponentType>
inline size_t Tensor<ComponentType>::numElements() const {
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
}

template<Arithmetic ComponentType>
inline const ComponentType &Tensor<ComponentType>::operator()(const std::vector<size_t> &idx) const {
    assert(idx.size() == _shape.size() || (idx.empty() && _shape[0] == 0));
    return _data[_calculateIndex(idx)];
}

template<Arithmetic ComponentType>
inline ComponentType &Tensor<ComponentType>::operator()(const std::vector<size_t> &idx) {
    assert(idx.size() == _shape.size() || (idx.empty() && _shape[0] == 0));
    return _data[_calculateIndex(idx)];
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::_calculateIndex(const std::vector<size_t> &idx) const {
    size_t index = std::inner_product(idx.begin(), idx.end(), _indexing.begin(), static_cast<size_t >(0));
    return index;
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::_calculateIndex(const std::vector<int> &idx) const {
    size_t index = std::inner_product(idx.begin(), idx.end(), _indexing.begin(), static_cast<size_t >(0));
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
inline bool operator==(const Tensor<ComponentType> &a, const Tensor<ComponentType> &b) {
    return a.checkEqual(b);
}

template<Arithmetic ComponentType>
inline bool Tensor<ComponentType>::checkEqual(const Tensor<ComponentType> &other) const {
    if (_shape != other._shape) {
        return false;
    }
    if (_data != other._data) {
        return false;
    }
    return true;
}

template<Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::readFromFile(const std::string &filename) {
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
    Tensor<ComponentType> tensor(shape);

    for (auto &element: tensor._data) {
        ifs >> element;
    }

    return tensor;
}

template<Arithmetic ComponentType>
void Tensor<ComponentType>::writeToFile(const std::string &filename) const {
    std::ofstream ofs(filename, std::ofstream::out);

    if (!ofs.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    ofs << this->rank() << std::endl;

    for (auto &element: _shape) {
        ofs << element << std::endl;
    }

    for (auto &element: _data) {
        ofs << element << std::endl;
    }
}

template<Arithmetic ComponentType>
void Tensor<ComponentType>::reshape(const std::vector<size_t> &newShape) {
    assert(static_cast<size_t>(std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>())) ==
           this->numElements());
    _shape = newShape;
    this->_fillIndexing();
}

template<Arithmetic ComponentType>
Iterator<ComponentType, typename std::vector<ComponentType>::iterator>
Tensor<ComponentType>::begin(const std::vector<int> &idx) {
    if (idx.empty()) {
        return Iterator<ComponentType, typename std::vector<ComponentType>::iterator>(_data.begin(), 1);
    }
    assert(idx.size() == _shape.size());
    assert(std::count(idx.begin(), idx.end(), -1) == 1);
    size_t position = std::distance(idx.begin(), std::find(idx.begin(), idx.end(), -1));

    auto it = _data.begin() + this->_calculateIndex(idx) + _indexing[position];
    return Iterator<ComponentType, typename std::vector<ComponentType>::iterator>(it,
                                                                                  static_cast<int>(_indexing[position]));
}

template<Arithmetic ComponentType>
Iterator<ComponentType, typename std::vector<ComponentType>::iterator>
Tensor<ComponentType>::end(const std::vector<int> &idx) {
    if (idx.empty()) {
        return Iterator<ComponentType, typename std::vector<ComponentType>::iterator>(_data.end(), 1);
    }
    assert(idx.size() == _shape.size());
    assert(std::count(idx.begin(), idx.end(), -1) == 1);
    size_t position = std::distance(idx.begin(), std::find(idx.begin(), idx.end(), -1));

    auto it =
            _data.begin() + this->_calculateIndex(idx) + _indexing[position] + (_indexing[position] * _shape[position]);
    return Iterator<ComponentType, typename std::vector<ComponentType>::iterator>(it,
                                                                                  static_cast<int>(_indexing[position]));
}

template<Arithmetic ComponentType>
Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>
Tensor<ComponentType>::begin(const std::vector<int> &idx) const {
    if (idx.empty()) {
        return Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>(_data.begin(), 1);
    }
    assert(idx.size() == _shape.size());
    assert(std::count(idx.begin(), idx.end(), -1) == 1);
    size_t position = std::distance(idx.begin(), std::find(idx.begin(), idx.end(), -1));

    auto it = _data.begin() + this->_calculateIndex(idx) + _indexing[position];
    return Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>(it,
                                                                                        static_cast<int>(_indexing[position]));
}

template<Arithmetic ComponentType>
Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>
Tensor<ComponentType>::end(const std::vector<int> &idx) const {
    if (idx.empty()) {
        return Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>(_data.end(), 1);
    }
    assert(idx.size() == _shape.size());
    assert(std::count(idx.begin(), idx.end(), -1) == 1);
    size_t position = std::distance(idx.begin(), std::find(idx.begin(), idx.end(), -1));

    auto it =
            _data.begin() + this->_calculateIndex(idx) + _indexing[position] + (_indexing[position] * _shape[position]);
    return Iterator<ComponentType, typename std::vector<ComponentType>::const_iterator>(it,
                                                                                        static_cast<int>(_indexing[position]));
}

template<Arithmetic ComponentType>
inline std::vector<ComponentType> &Tensor<ComponentType>::data() {
    return _data;
}

template<Arithmetic ComponentType>
inline const std::vector<ComponentType> &Tensor<ComponentType>::data() const {
    return _data;
}

// Pretty-prints the tensor to stdout.
// This is not necessary (and not covered by the tests) but nice to have, also for debugging (and for exercise of course...).
template<Arithmetic ComponentType>
std::ostream &
operator<<(std::ostream &out, const Tensor<ComponentType> &tensor) {

    if (tensor.rank() == 0) {
        return out;
    }

    std::vector<size_t> shape = tensor.shape();

    out << "shape:(";
    for (size_t element: shape) {
        out << element << ", ";
    }
    out << '\b' << '\b';
    out << ")\n";

    std::vector<size_t> multipliers;
    for (size_t offset = 0; offset < shape.size(); offset++) {
        multipliers.push_back(std::accumulate(shape.begin() + offset, shape.end(), 1, std::multiplies<>()));
    }
    std::reverse(multipliers.begin(), multipliers.end());
    size_t index = 0;
    bool open_line = false;

    out << std::string(shape.size(), '[');

    for (auto it = tensor.begin({}); it != tensor.end({}); it++, index++) {
        out << *it << ", ";

        for (size_t mindex = 0; mindex < multipliers.size() - 1; mindex++) {

            if ((index + 1) % multipliers[mindex] == 0 && (index + 1) % multipliers[mindex + 1] != 0) {
                if (!open_line)
                    out << '\b' << '\b' << "]\n" << std::string(shape.size() - 1, ' ') << "[";
                else {
                    out << '\b' << '\b' << "]" << std::string(mindex + 1, '\n') << ' '
                        << std::string(shape.size() - (mindex + 2), ' ') << std::string(mindex + 1, '[');
                    open_line = false;
                }
            }
            if ((index + 1) % multipliers[mindex + 1] == 0) {
                out << '\b' << '\b' << "]  ";
                open_line = true;
            }
        }
    }

    out << '\b' << '\b' << "]\n";
    return out;


}

// Reads a tensor from file.
template<Arithmetic ComponentType>
Tensor<ComponentType> readTensorFromFile(const std::string &filename) {
    return Tensor<ComponentType>::readFromFile(filename);
}


template<Arithmetic ComponentType>
void writeTensorToFile(const Tensor<ComponentType> &tensor, const std::string &filename) {
    tensor.writeToFile(filename);
}
