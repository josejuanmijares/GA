module DummyModule

	export MyType, f

	type MyType	
		#a::Array{Int64,1}
		a::SharedArray
		fu::Function
		f2::Function
		
		function MyType()
			this = new()
			this.a = zeros(Int64,20)
			this.fu = function(k)
				this.a[k]=k
			end
			
			this.f2 = function(k2)
				pmap(this.fu,[1:k2;])
			end
			
			return this
		end
			
	end

	f(x) = x^2+1

	println("loaded")

end