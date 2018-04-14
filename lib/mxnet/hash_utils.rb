
unless Hash.new.respond_to? :transform_keys
  class Hash
    # File activesupport/lib/active_support/core_ext/hash/keys.rb, line 13
    def transform_keys
      return enum_for(:transform_keys) { size } unless block_given?
      result = {}
      each_key do |key|
        result[yield(key)] = self[key]
      end
      result
    end
  end
end
