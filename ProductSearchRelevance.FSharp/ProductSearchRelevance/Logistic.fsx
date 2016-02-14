(*
String manipulation
*)

open System.Text.RegularExpressions

let lower (s:string) = s.ToLowerInvariant ()

let matchWords = Regex(@"\w+", RegexOptions.Compiled)

let wordTokenizer (text:string) =
    text
    |> matchWords.Matches
    |> Seq.cast<Match>
    |> Seq.map (fun m -> m.Value)
    |> Set.ofSeq

let normalize = lower

(*
Prelude: types
*)

#I "../packages"
#r @"FSharp.Data/lib/net40/FSharp.Data.dll"

open FSharp.Data

[<Literal>]
let trainPath = @"../data/train.csv"
[<Literal>]
let testPath = @"../data/test.csv"
[<Literal>]
let attributesPath = @"../data/attributes.csv"
[<Literal>]
let productsPath =  @"../data/product_descriptions.csv"

type Train = CsvProvider<trainPath,Schema=",,,,float">
type Test = CsvProvider<testPath>
type Attributes = CsvProvider<attributesPath>
type Products = CsvProvider<productsPath>

type UID = int
type Query = string
type SearchResult = Query * UID

let productDescriptions =

    printfn "Extracting product descriptions"

    Products.GetSample().Rows
    |> Seq.map (fun x ->
        x.Product_uid,
        x.Product_description)
    |> Map.ofSeq

let attributes =

    printfn "Extracting attributes"
    Attributes.GetSample().Rows
    |> Seq.map (fun x ->
        x.Name |> normalize,
        x.Value |> normalize)
    |> Seq.groupBy fst
    |> Seq.map (fun (key,values) ->
        key,
        values |> Seq.map snd |> set)
    |> Map.ofSeq

let productAttributes =

    printfn "Extracting product attributes"
    Attributes.GetSample().Rows
    |> Seq.map (fun att ->
        let uid = att.Product_uid
        let attribute = att.Name |> normalize
        let value = att.Value |> normalize
        uid, attribute, value)
    |> Seq.groupBy (fun (uid,att,value) -> uid)
    |> Seq.map (fun (uid,group) ->
        uid,
        group
        |> Seq.map (fun (uid,att,value) -> att,value)
        |> Map.ofSeq)
    |> Map.ofSeq

(*
Features construction
*)

let brandAttribute = "MFG Brand Name" |> normalize

let brandNames = attributes.[brandAttribute]

let brandOf uid =
    productAttributes
    |> Map.tryFind uid
    |> Option.bind (fun productAttributes ->
        productAttributes.TryFind brandAttribute)

let brandsInString (s:string) =
    let tokens = s |> wordTokenizer
    Set.intersect brandNames tokens


let featurize (query:Query,uid:UID) =

    let q = normalize query
    let tokens = wordTokenizer q

    let exactMatches =
        brandNames
        |> Set.filter (q.Contains)
        |> Set.count

    //let noMatch = if exactMatches = 0 then 1. else 0.
    let oneExactMatch = if exactMatches = 1 then 1. else 0.
    let manyMatches = if exactMatches > 1 then 1. else 0.
    let queryLength = q.Length |> float
    let wordsLength = wordTokenizer q |> Set.count |> float

    let brand = brandOf uid
    let brandMatch =
        match brand with
        | None -> 0.
        | Some(name) ->
            if q.Contains name then 1. else 0.

    let brandMismatch =
        match brand with
        | None -> 0.
        | Some(name) ->
            if q.Contains name then 0. else 1.

    [|
        oneExactMatch
        manyMatches
        queryLength
        brandMatch
        brandMismatch
        wordsLength
    |]

let scale relevance = (relevance - 1.) / 2.
let descale output = (output * 2.) + 1.

(*
Training
*)

#r "Accord/lib/net45/Accord.dll"
#r "Accord.Math/lib/net45/Accord.Math.dll"
#r "Accord.Statistics/lib/net45/Accord.Statistics.dll"

open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting


type Features = float[]
type Label = float

let train (features:Features[]) (labels:Label []) =

    let numberOfFeatures = 6
    let regression = LogisticRegression(numberOfFeatures)
    let learner = IterativeReweightedLeastSquares(regression)

    let rec refine () =
        let delta = learner.Run(features, labels)
        printfn "%.6f" delta
        if delta < 0.0001
        then learner
        else refine ()

    let model = refine ()

    // diagnostics
    model.ComputeError(features,labels) |> printfn "Error: %.3f"

    [ 0 .. (numberOfFeatures - 1)]
    |> List.iter (fun i ->
        printfn "Feat. %i: coeff %.5f sign %b"
            i
            regression.Coefficients.[i]
            (regression.GetWaldTest(i).Significant))

    // return predictor
    fun (x:Features) -> x |> regression.Compute |> descale

(*
Validation
*)

let trainingSet = Train.GetSample().Rows |> Seq.toArray
let size = trainingSet.Length

let features, labels =
    trainingSet
    |> Array.map (fun x ->
        featurize (x.Search_term, x.Product_uid),
        (float x.Relevance) |> scale)
    |> Array.unzip

type Quality = float

let rmse (actual:Quality seq) (expected:Quality seq) =
    Seq.zip actual expected
    |> Seq.averageBy (fun (act,exp) ->
        let delta = act - exp
        delta * delta)
    |> sqrt

let validate (features:Features[]) (labels:Label []) model =

    let predicted = features |> Array.map model
    let actual = labels

    rmse predicted actual |> printfn "Validation: RMSE %.3f"

let trainSize = trainingSet.Length * 3 / 4

let trainFeatures, testFeatures = features.[..trainSize], features.[trainSize+1..]
let trainLabels, testLabels = labels.[..trainSize], labels.[trainSize+1..]

train trainFeatures trainLabels
|> validate testFeatures testLabels
|> ignore

(*
Prepare submission file
*)

let createSubmission () =

  let testSample = Test.GetSample ()
  let testSize = testSample.Rows |> Seq.length

  open System.IO

  let submission =
      [|
          yield "id,relevance"
          for case in testSample.Rows ->
              let fs = featurize (case.Search_term, case.Product_uid)
              let predicted = regression.Compute fs |> denorm
              sprintf "%i,%.2f" (case.Id) predicted
      |]

  let submissionPath = @"C:/Users/Mathias Brandewinder/Desktop/submission.csv"

  File.WriteAllLines(submissionPath,submission)
