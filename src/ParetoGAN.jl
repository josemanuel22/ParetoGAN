module ParetoGAN

using Flux
using Distributions
using StatsBase
using Distances

export ParetoGAN, train!, generate

# ——— Tail‐index estimation as before ———
function estimate_tail_index(X::AbstractMatrix; k::Int = ceil(Int, size(X, 1) * 0.1))
	n, d = size(X)
	ξ = zeros(d)
	for j in 1:d
		col = sort(X[:, j]; rev = true)
		topk = col[1:k]
		x_k = topk[end]
		ξ[j] = mean(log.(topk ./ x_k))
	end
	return ξ
end

# ——— Generator builder ———
function build_generator(in_dim::Int, hidden::Vector{Int}, out_dim::Int)
	layers = Chain(Dense(in_dim, hidden[1], relu))
	for i in 1:length(hidden)-1
		push!(layers.layers, Dense(hidden[i], hidden[i+1], relu))
	end
	push!(layers.layers, Dense(hidden[end], out_dim))
	return layers
end

# ——— Discriminator builder ———
function build_discriminator(in_dim::Int, hidden::Vector{Int})
	layers = Chain(Dense(in_dim, hidden[1], leakyrelu; α = 0.2))
	for i in 1:length(hidden)-1
		push!(layers.layers, Dense(hidden[i], hidden[i+1], leakyrelu; α = 0.2))
	end
	push!(layers.layers, Dense(hidden[end], 1), σ)  # output in (0,1)
	return layers
end

# ——— ParetoGAN type holds G, D, tail‐indices, etc. ———
struct ParetoGAN
	gen::Chain
	dis::Chain
	ξ::Vector{Float64}
	β::Union{Float64, Vector{Float64}}
end

"""
	ParetoGAN(X; g_hidden=[128,128], d_hidden=[128,128], β=1.0, k=…)

Builds a GAN for data `X`. Learns tail‐indices ξ, constructs a generator
and a discriminator with the specified hidden layers.
"""
function ParetoGAN(X::AbstractMatrix;
	g_hidden = [128, 128],
	d_hidden = [128, 128],
	β = 1.0,
	k = ceil(Int, size(X, 1) * 0.1))
	ξ = estimate_tail_index(X; k = k)
	d = size(X, 2)
	G = build_generator(length(ξ), g_hidden, d)
	D = build_discriminator(d, d_hidden)
	return ParetoGAN(G, D, ξ, β)
end

# ——— Sample noise from fitted GPD ———
rand_noise(pgan::ParetoGAN, n::Int) = begin
	d = length(pgan.ξ)
	Z = [rand(GeneralizedPareto(pgan.ξ[i], pgan.β)) for _ in 1:n, i in 1:d]
	reshape(Z, n, d)
end

# ——— Standard BCE losses for D and G ———
function d_loss(pgan::ParetoGAN, X_real, Z)
	X_fake = pgan.gen(Z)
	r = pgan.dis(X_real')  # Flux expects features × batch
	f = pgan.dis(X_fake')
	loss = Flux.Losses.binarycrossentropy.(r, ones(size(r))) .+
		   Flux.Losses.binarycrossentropy.(f, zeros(size(f)))
	return mean(loss)
end

function g_loss(pgan::ParetoGAN, Z)
	X_fake = pgan.gen(Z)
	f = pgan.dis(X_fake')
	return mean(Flux.Losses.binarycrossentropy.(f, ones(size(f))))
end

"""
	train!(pgan, X_real; epochs=100, batch=64, opt_g=ADAM(), opt_d=ADAM())

Adversarial training: for each batch, first update D, then G using Flux.withgradient.
"""
function train!(pgan::ParetoGAN, X_real; epochs = 100, batch = 64,
	opt_g = ADAM(1e-4), opt_d = ADAM(1e-4))
	N, _ = size(X_real)
	for epoch in 1:epochs
		for idx in Iterators.partition(1:N, batch; shuffle = true)
			Xb = X_real[idx, :]
			# sample noise
			Zb = rand_noise(pgan, length(idx))

			# —– Discriminator step —–
			d_val, back_d = Flux.withgradient(pgan.dis) do D
				d_loss(ParetoGAN(pgan.gen, D, pgan.ξ, pgan.β), Xb, Zb)
			end
			gs_d = back_d()
			Flux.Optimise.update!(opt_d, Flux.params(pgan.dis), gs_d)

			# —– Generator step —–
			g_val, back_g = Flux.withgradient(pgan.gen) do G
				g_loss(ParetoGAN(G, pgan.dis, pgan.ξ, pgan.β), Zb)
			end
			gs_g = back_g()
			Flux.Optimise.update!(opt_g, Flux.params(pgan.gen), gs_g)
		end

		@info "Epoch $epoch — D_loss=$(d_val) — G_loss=$(g_val)"
	end
	return pgan
end

"""
	generate(pgan, n)

Produce `n` samples by passing GPD‐noise through the trained generator.
"""
function generate(pgan::ParetoGAN, n::Int)
	Z = rand_noise(pgan, n)
	return pgan.gen(Z)
end

end
