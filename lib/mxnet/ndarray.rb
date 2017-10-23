module MXNet
  class NDArray
    def self.ones(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      case dtype
      when String, Symbol
        dtype = MXNet::DType.name2id(dtype)
      when Integer
        # do nothing
      else
        raise TypeError, "wrong type of dtype #{dtype.class} (expected Symbol or Integer)"
      end
      Ops._ones(shape, ctx, dtype, **kwargs)
    end

    def self.zeros(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      case dtype
      when String, Symbol
        dtype = MXNet::DType.name2id(dtype)
      when Integer
        # do nothing
      else
        raise TypeError, "wrong type of dtype #{dtype.class} (expected Symbol or Integer)"
      end
      Ops._zeros(shape, ctx, dtype, **kwargs)
    end

    # Returns a sliced view of this array.
    #
    # @param [Integer, Range, Array] key  Indexing key.
    # @return [NDArray] a sliced view of this array.
    def [](key)
      case key
      when Integer
        if key > shape[0] - 1
          raise IndexError, "index #{key} is out of bounds for axis 0 with size #{shape[0]}"
        end
        _at(key)
      when Range
        if key.begin || key.end
          _slice(key.begin, key.end)
        else
          self
        end
      when Array
        keys = key
        shape = self.shape
        unless shape.length >= keys.length
          raise IndexError, "Slicing dimensions exceeds array dimensions, #{keys.length} vs #{shape.length}"
        end
        out_shape, begins, ends = [], [], []
        keys.each_with_index do |key, idx|
          case key
          when Integer
            begins << key
            ends << key + 1
          when Range
            begins << (key.begin.nil? || key.begin == -Float::INFINITY) ? 0 : key.begin
            ends << (key.end.nil? || key.end == Float::INFINITY) ? shape[i] : key.end
            out_shape << ends.last - begins.last
          else
            raise IndexError, "NDArray does not support slicing with key #{key} of type #{key.class}"
          end
        end
        out_shape.concat(shape[keys.length..-1])
        out_shape << 1 if out_shape.empty?
        return slice(begins, ends).reshape(out_shape)
      else
        raise IndexError, "NDArray does not support slicing with key #{key} of type #{key.class}"
      end
    end

    # Set self[key] to value.
    #
    # @param [nil, Integer, Range, or Array] key  The indexing key.
    # @param [scalar, NDArray, or Array] value  The value to set.
    # @return the given value
    def []=(key, value)
      case key
      when Integer
        self[key][nil] = value
      when Range
        # TODO: Support step range
        self[key][nil] = value
      when Array
      when nil
        case value
        when NDArray
          value.copyto(self) if value.handle != self.handle
        when Integer, Float
          _set_value(value, self)
        when Array
          _sync_copyfrom(value)
        else
          raise ArgumentError, "NDArray does not support assignment with #{value} of type #{value.class}"
        end
      end
    end

    def ndim
      shape.length
    end
    alias rank ndim

    def size
      shape.inject(:*)
    end
    alias length size

    def transpose(axes=nil)
      Ops.transpose(self, axes)
    end

    def as_scalar
      unless shape == [1]
        raise TypeError, "The current array is not a scalar"
      end
      to_a[0]
    end

    def +(other)
      case other
      when NDArray
        Ops.broadcast_add(self, other)
      else
        super
      end
    end

    def -(other)
      case other
      when NDArray
        Ops.broadcast_sub(self, other)
      else
        super
      end
    end

    def *(other)
      case other
      when NDArray
        Ops.broadcast_mul(self, other)
      else
        super
      end
    end

    def /(other)
      case other
      when NDArray
        Ops.broadcast_div(self, other)
      else
        super
      end
    end

    module Ops
      def self._import_ndarray_operations
        LibMXNet._each_op_names do |op_name|
          op_handle = LibMXNet._get_op_handle(op_name)
          op_info = LibMXNet._get_op_info(op_handle)
        end
      end
    end
  end

  def self.NDArray(source_array, ctx=nil, dtype=:float32, **kwargs)
    case source_array
    when NDArray
      shape = source_array.shape
    else
      dtype ||= :float32
      source_array = source_array.to_ary
      shape, = discover_dimensions(source_array)
    end
    array = NDArray.empty(shape)
    array[0..-1] = source_array
    array
  end

  private

  MAXDIMS = 32  # This value is from NPY_MAXDIMS

  def self.discover_dimensions(ary, maxndim=MAXDIMS)
    return [[], 0] if maxndim == 0 || !ary.is_a?(Array)

    shape = [ary.length]
    return [shape, 1] if ary.length == 0 || maxndim == 1

    sub_shape, maxndim_m1 = discover_dimensions(ary[0], maxndim - 1)
    shape.concat(sub_shape)
    maxndim = maxndim_m1 + 1

    (1 ... ary.length).each do |i|
      sub_shape, maxndim_m1 = discover_dimensions(ary[i], maxndim - 1)
      (0 ... maxndim_m1).each do |j|
        unless sub_shape[j] == shape[j+1]
          maxndim_m1 = j
          break
        end
      end
    end
    raise ArgumentError, "Array has inconsistent dimensions" if maxndim_m1 + 1 < maxndim

    [shape, maxndim]
  end
end
