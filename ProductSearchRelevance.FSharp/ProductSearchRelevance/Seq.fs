namespace HomeDepot

open System
open System.Collections
open System.Collections.Generic
open System.Diagnostics

module Seq =
    type private TimedEnumerator<'t>(source : IEnumerator<'t>, blockSize, f) =
        let swUpstream = Stopwatch()
        let swDownstream = Stopwatch()
        let swTotal = Stopwatch()
        let mutable count = 0
        interface IEnumerator<'t> with
            member __.Reset() = failwith "not implemented"
            member __.Dispose() = source.Dispose()
            member __.Current = source.Current
            member __.Current = (source :> IEnumerator).Current
            member __.MoveNext() =
                // downstream consumer has finished processing the previous item
                swDownstream.Stop()

                // start the total throughput timer if this is the first item
                if not swTotal.IsRunning then
                    swTotal.Start()

                // measure upstream MoveNext() call
                swUpstream.Start()
                let result = source.MoveNext()
                swUpstream.Stop()

                if result then
                    count <- count + 1

                // blockSize reached or upstream is complete - invoke callback and reset timers
                if (result && count = blockSize) || (not result && count > 0) then
                    swTotal.Stop()
                    let upDownTotal = swUpstream.Elapsed + swDownstream.Elapsed
                    f count swTotal.Elapsed (swUpstream.Elapsed.TotalMilliseconds / upDownTotal.TotalMilliseconds)
                    count <- 0
                    swUpstream.Reset()
                    swDownstream.Reset()
                    swTotal.Restart()

                // clock starts for downstream consumer
                swDownstream.Start()
                result

    type private SlimTimedEnumerator<'t>(source : IEnumerator<'t>, blockSize, f) =
        let swTotal = Stopwatch()
        let mutable count = 0
        interface IEnumerator<'t> with
            member __.Current = source.Current
            member __.Current = (source :> IEnumerator).Current
            member __.Reset() = failwith "not implemented"
            member __.Dispose() = source.Dispose()
            member __.MoveNext() =
                // start the timer on the first MoveNext call
                if not swTotal.IsRunning then
                    swTotal.Start()

                let result = source.MoveNext()
                if result then
                    count <- count + 1

                // blockSize reached or upstream is complete - invoke callback and reset timer
                if (result && count = blockSize) || (not result && count > 0) then
                    swTotal.Stop()
                    f count swTotal.Elapsed Double.NaN
                    count <- 0
                    swTotal.Restart()
                result

    let mkTimed blockSize getEnum =
        let block =
            match blockSize with
            | Some(bs) when bs <= 0 -> invalidArg "blockSize" "blockSize must be positive"
            | Some(bs) -> bs
            | None -> -1
        { new IEnumerable<'a> with
              member __.GetEnumerator() = upcast (getEnum block)
          interface IEnumerable with
              member __.GetEnumerator() = upcast (getEnum block) }

    let timed blockSize f (source : seq<'t>) =
        mkTimed blockSize (fun block -> new TimedEnumerator<'t>(source.GetEnumerator(), block, f))

    let timedSlim blockSize f (source : seq<'t>) =
        mkTimed blockSize (fun block -> new SlimTimedEnumerator<'t>(source.GetEnumerator(), block, f))

// sample callbacks that scale time units
module Scale =
    let private stats unitString getUnit blockSize elapsedTotal upstreamRatio =
        let totalScaled = getUnit (elapsedTotal : TimeSpan)
        let itemsPerTime = (float blockSize) / totalScaled
        let timePerItem = totalScaled / (float blockSize)
        let upDownPercentStr =
            if Double.IsNaN(upstreamRatio) then ""
            else sprintf " (%2.0f%% upstream | %2.0f%% downstream)" (100. * upstreamRatio) (100. * (1. - upstreamRatio))

        Console.WriteLine(
            sprintf "%d items %10.2f{0} %10.2f items/{0} %10.2f {0}/item%s"
                blockSize totalScaled itemsPerTime timePerItem upDownPercentStr,
            (unitString : string)
        )

    let s = stats "s" (fun ts -> ts.TotalSeconds)
    let ms = stats "ms" (fun ts -> ts.TotalMilliseconds)
    let μs = stats "μs" (fun ts -> ts.TotalMilliseconds * 1e3)
    let ns = stats "ns" (fun ts -> ts.TotalMilliseconds * 1e6)
