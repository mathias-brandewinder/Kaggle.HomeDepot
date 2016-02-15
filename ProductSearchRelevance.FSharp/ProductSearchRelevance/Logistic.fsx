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

type Observation = {
    Product_uid:UID
    Search_term:Query
    ID:int }

type Relevance = float
type Example = Observation * Relevance

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

(*
Training
*)

#r @"Charon/lib/Charon.dll"
open Charon

let labels = 
    "Relevance", 
    (fun (relevance:float) -> 
        if relevance < 1.5 then "bad"
        elif relevance > 2.5 then "good"
        else "average"
        |> Some) |> Categorical

(*
fine-grain labels
*)

let fineLabels = 
    "Relevance", 
    (fun (relevance:float) -> 
        if relevance = 1.0 then "bad"
        elif relevance = 3.0 then "excellent"
        elif relevance <= 1.5 then "mediocre"
        elif relevance >= 2.5 then "good"
        else "average"
        |> Some) |> Categorical

let wordsInQuery = 
    "Words", 
    (fun (o:Observation) -> 
        o.Search_term 
        |> wordTokenizer
        |> Set.count
        |> fun count ->
            if count = 1 then "1"
            elif count = 2 then "2"
            elif count = 3 then "3"
            elif count = 4 then "4"
            elif count = 5 then "5"
            else "many"
        |> Some)
        |> Categorical

let brandMatch =
    "Brand match",
    (fun (o:Observation) -> 
        let brand = brandOf (o.Product_uid)
        match brand with
        | None -> "N/A"
        | Some(name) ->
            if (normalize o.Search_term).Contains name 
            then "match" 
            else "no match"
        |> Some)
        |> Categorical

let numberOfAttributes =
    "Number of attributes",
    (fun (o:Observation) -> 
        let attributes = 
            productAttributes.TryFind (o.Product_uid)
        match attributes with
        | None -> "N/A"
        | Some(attributes) ->
            let count = attributes.Count
            if count < 10 then "low"
            elif count < 20 then "med"
            else "hi"
        |> Some)
        |> Categorical

let bullet1 = "bullet01"

let firstBulletMatch =
    "bullet1 match",
    (fun (o:Observation) -> 
        let atts = productAttributes.TryFind o.Product_uid
        match atts with
        | None -> "No attributes"
        | Some(atts) ->
            let b1 = atts.TryFind bullet1
            match b1 with
            | None -> "N/A"
            | Some(bullet) ->
                let desc = bullet |> wordTokenizer
                let query = o.Search_term |> wordTokenizer
                if query.Count < 3 then
                    if Set.intersect desc query = query
                    then "match" 
                    else "no match"
                else
                    if ((Set.intersect desc query).Count >= query.Count - 1) 
                    then "match" 
                    else "no match"
        |> Some)
        |> Categorical
    
let descriptionMatch =
    "description match",
    (fun (o:Observation) -> 
        let description = productDescriptions.TryFind o.Product_uid
        match description with
        | None -> "No desc"
        | Some(text) ->
            let desc = text |> wordTokenizer
            let query = o.Search_term |> wordTokenizer
            if query.Count < 3 then
                if Set.intersect desc query = query
                then "match" 
                else "no match"
            else
                if ((Set.intersect desc query).Count >= query.Count - 1) 
                then "match" 
                else "no match"
        |> Some)
        |> Categorical

let kit =
    "kit",
    (fun (o:Observation) -> 
        if (o.Search_term |> normalize).Contains "kit"
        then "kit"
        else "no kit"
        |> Some)
        |> Categorical

let features = 
    [
        wordsInQuery
        brandMatch
        numberOfAttributes
        firstBulletMatch
        descriptionMatch
        kit
    ]

let data = 
    Train.GetSample().Rows 
    |> Seq.map (fun row -> 
        row.Relevance,
        {   ID = row.Id
            Product_uid = row.Product_uid 
            Search_term = row.Search_term })
    |> Seq.toArray

(* 
Experimentations 
*)

let training = data

let tree = basicTree training (labels,features) { DefaultSettings with Holdout = 0.2 }

tree.Pretty |> printfn "%s"

tree.TrainingQuality
tree.HoldoutQuality

let forestResults = forest training (labels,features) { DefaultSettings with ForestSize = 20 }
printfn "OOB quality: %f" forestResults.OutOfBagQuality

(*
Validation
*)

let rmse (actual:Relevance seq) (expected:Relevance seq) =
    Seq.zip actual expected
    |> Seq.averageBy (fun (act,exp) ->
        let delta = act - exp
        delta * delta)
    |> sqrt

let size = training.Length
let trainSize = size * 3 / 4

let model = 
    //basicTree training.[..trainSize] (labels,features) { DefaultSettings with Holdout = 0.0 }
    forest training.[..trainSize] (fineLabels,features) { DefaultSettings with ForestSize = 20 }

//training 
//|> Seq.countBy fst 
//|> Seq.toList 
//|> List.sortBy fst

let numerize output = 
    if output = "bad" then 1.33 
    elif output = "good" then 2.67
    else 2.33

let numerize2 output = 
        if output = "bad" then 1.0 
        elif output = "excellent" then 3.0
        elif output = "mediocre" then 1.67  
        elif output = "good" then 2.67  
        else 2.33

let validate model =

    let predicted = 
        training.[trainSize+1..]
        |> Array.map (snd >> model >> numerize2)
    let actual =
        training.[trainSize+1..]
        |> Array.map fst
    rmse predicted actual |> printfn "RMSE: %.2f"

validate (model.Classifier)

(*
Prepare submission file
*)

open System.IO

printfn "Training..."
let fullModel = 
    forest training (fineLabels,features) { DefaultSettings with ForestSize = 100 }

printfn "Preparing predictions..."

let createSubmission () =

  let testSample = 
    Test.GetSample().Rows
    |> Seq.map (fun row -> 
        {   ID = row.Id
            Product_uid = row.Product_uid 
            Search_term = row.Search_term })

  let predictor = fullModel.Classifier

  let submission =
      [|
          yield "id,relevance"
          for case in testSample ->
              let predicted = predictor case |> numerize2
              sprintf "%i,%.2f" (case.ID) predicted
      |]

  let submissionPath = @"C:/Users/Mathias Brandewinder/Desktop/submission.csv"

  File.WriteAllLines(submissionPath,submission)

createSubmission ()