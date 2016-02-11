﻿#I @"../packages/"

#r "Accord/lib/net45/Accord.dll"
#r "FSharp.Data/lib/net40/FSharp.Data.dll"
#r "Accord.Math/lib/net45/Accord.Math.dll"
#r "Accord.Statistics/lib/net45/Accord.Statistics.dll"
#r "FSharp.Collections.ParallelSeq/lib/net40/FSharp.Collections.ParallelSeq.dll"
#r "StemmersNet/lib/net20/StemmersNet.dll"
#r "alglibnet2/lib/alglibnet2.dll"
#r "FuzzyString/lib/FuzzyString.dll"
#load "CsvData.fs"
#load "StringUtils.fs"
#load "Core.fs"
#load "TFIDF.fs"

open System
open System.Text.RegularExpressions
open FSharp.Collections.ParallelSeq
open HomeDepot.Core
open StringUtils
open System.Text

printfn "Building Brand Name set..."
let brands =
    CsvData.attributes.Rows
    |> Seq.where (fun r -> r.Name = "MFG Brand Name")
    |> Seq.map (fun r -> r.Value.ToLowerInvariant())
    |> Set.ofSeq

let brandName attribMap uid =
    match attribMap |> Map.tryFind uid with
    | Some a ->
      let brand = a |> Seq.tryFind (fun (_, name, _) -> name = "MFG Brand Name")
      brand |> Option.map (fun (_, _, value) -> value)
    | None -> None

let isMatch input word =
    let word' = word |> sanitize |> stem |> Regex.Escape
    Regex.IsMatch(input |> sanitize |> stemWords, sprintf @"\b%s" word', RegexOptions.IgnoreCase)

let fixLineConcats (attribs:string seq) (desc:string) =
    attribs
    |> Seq.where (fun a -> not <| String.IsNullOrWhiteSpace a)
    |> Seq.fold
        (fun (state:StringBuilder) t -> state.Replace(t, t + " "))
        (StringBuilder(desc))
    |> string

let features attrSelector productBrand (sample:CsvData.Sample) =
    let words = sample.Query
    let uniqueWords = words |> splitOnSpaces |> Array.distinct
    let title = sample.Title
    let titleWords = title |> splitOnSpaces
    let indices = uniqueWords |> Seq.map (fun w -> titleWords |> Seq.tryFindIndexBack (isMatch w)) |> Seq.choose id
    let queryTitleIndices = indices |> Seq.map (fun i -> float i / float titleWords.Length) |> Array.ofSeq
    let queryTitlePosScore = if Array.isEmpty queryTitleIndices then 0. else Seq.average queryTitleIndices
    let attributes = attrSelector sample.ProductId
    let desc = sample.Description |> fixLineConcats attributes
    let deduped = attributes |> Seq.where (fun a -> a |> containedIn desc |> not) |> String.concat " "
    let titleMatches = uniqueWords |> Array.filter (isMatch title)
    let descMatches = uniqueWords |> Array.filter (isMatch desc)
    let attrMatches = uniqueWords |> Array.filter (isMatch deduped)
    let wordMatchCount =
        uniqueWords
        |> Seq.filter (fun w -> Seq.concat [titleMatches; descMatches; attrMatches] |> Seq.contains w)
        |> Seq.length
    let brandNameMatch =
        match productBrand with // does query contain product brand?
        | Some bn -> if uniqueWords |> Array.exists (containedIn bn) then 1 else 0
        | None ->
          // does query contain any brand name?
          let searchedBrand = brands |> Set.filter (containedIn words) |> Seq.tryHead
          match searchedBrand with // is query brand name in product title?
          | Some b -> if b |> containedIn title then 1 else -1
          | None -> 0
    [| float uniqueWords.Length
       float words.Length
       float title.Length
       float desc.Length
       float wordMatchCount
       float titleMatches.Length
       float titleMatches.Length / float uniqueWords.Length
       float descMatches.Length
       float descMatches.Length / float uniqueWords.Length
       float attrMatches.Length
       queryTitlePosScore
//       float attrMatches.Length / float uniqueWords.Length
       float brandNameMatch |]

let getAttr attribMap productId =
    match attribMap |> Map.tryFind productId with
    | Some a ->
      let getAttrStr name (value:string) =
          match value.ToLowerInvariant() with
          | "yes" -> name // if true attrib, include attrib name
          | "no"  -> String.Empty
          | _     -> name + " " + value
      a |> Seq.map (fun (_, name, value) -> getAttrStr name value) |> Array.ofSeq
    | None -> [||]

let getFeatures attribs attribMap sample =
    sample |> features attribs (brandName attribMap sample.ProductId)

let extractFeatures featureExtractor = 
    PSeq.ordered
    >> PSeq.map featureExtractor
    >> PSeq.toArray

let rfLearn (examples:Example array) attribMap =
  let samples, trainOutput = Array.unzip examples

  printfn "Extracting training features..."
  let attribs = getAttr attribMap
  let getFeatures' = getFeatures attribs attribMap
  let trainInput = samples |> extractFeatures getFeatures'
  // NOTE: ALGLIB wants prediction variable at end of input array
  let trainInputOutput =
      Seq.zip trainInput trainOutput
      |> Seq.map (fun (i,o) -> Array.append i [|o|])
      |> array2D

  printfn "Random Decision Forest regression..."
  let trees = 400
  let treeTrainSize = 0.05
  let featureCount = trainInput.[0].Length
  let _info, forest, forestReport =
      alglib.dfbuildrandomdecisionforest(trainInputOutput, trainInput.Length, featureCount, 1, trees, treeTrainSize)
  printfn "RDF RMS Error: %f; Out-of-bag RMS Error: %f" forestReport.rmserror forestReport.oobrmserror

  let predict samples =
    let trainInput = samples |> extractFeatures getFeatures'
    let mutable result : float [] = [||]
    trainInput
    |> Array.map (fun features -> 
        alglib.dfprocess(forest, features, &result)
        result.[0])
  predict

//let rfQuality = evaluate rfLearn
submission rfLearn
//0.48737 = kaggle rsme; oobrmserror = 0.4776784128; rmserror = 0.4303968628
//? = kaggle rsme; oobrmserror = 0.4147019175; rmserror = 0.3529753185
//0.48371 = kaggle rsme; RDF RMS Error: 0.451875; Out-of-bag RMS Error: 0.475592