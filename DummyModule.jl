module DummyModule

	export MyType, f

	type MyType	
		a::Int
		fu::Function
		f2::Function
		
		function MyType()
			this = new()
			
			this.fu = function(k)
				println(k)
			end
			
			this.f2 = function(k2)
				return pmap(this.fu,[1:k2;])
			end
			
			return this
		end
			
	end

	f(x) = x^2+1

	println("loaded")

end