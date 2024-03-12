using DrWatson, Test
@quickactivate "ABoxWorld"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "ABoxWorld tests" begin
    @test 1 == 1
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")

#Logging in DrWatson:
using Dates

function logmessage(n, error)
    # current time
    time = Dates.format(now(UTC), dateformat"yyyy-mm-dd HH:MM:SS")

    # memory the process is using
    maxrss = "$(round(Sys.maxrss()/1048576, digits=2)) MiB"

    logdata = (;
        n, # iteration n
        error, # some super important progress update
        maxrss) # lastly the amount of memory being used

    println(savename(time, logdata; connector=" | ", equals=" = ", sort=false, digits=2))
end

function expensive_computation(N)
    for n = 1:N
        sleep(1) # heavy computation
        error = rand()/n # some super import progress update
        logmessage(n, error)
    end
end

expensive_computation(5)