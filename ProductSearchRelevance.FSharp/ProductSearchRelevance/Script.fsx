#I @"../packages/"
#r "FSharp.Data/lib/net40/FSharp.Data.dll"
#r @"../Harness/bin/Debug/Harness.dll"

open HomeDepot.Model

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
Features construction
*)

let brandAttribute = "MFG Brand Name"

let brandNames = attributes.[brandAttribute]

let brandOf (product:Product) =
    product.Attributes.TryFind brandAttribute

let brandsInString (s:string) =
    let tokens = s |> wordTokenizer
    Set.intersect brandNames tokens

#r @"Charon/lib/Charon.dll"
open Charon

(*
simplified labels
*)

let labels = 
    "Relevance", 
    (fun (relevance:float) -> 
        if relevance < 1.5 then "bad"
        elif relevance > 2.5 then "good"
        else "average"
        |> Some) |> Categorical

let numerize output = 
    if output = "bad" then 1.33 
    elif output = "good" then 2.67
    else 2.33

let wordsInQuery = 
    "Words", 
    (fun (o:Observation) -> 
        o.SearchTerm 
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
        let brand = brandOf (o.Product)
        match brand with
        | None -> "N/A"
        | Some(name) ->
            let normalizedName = normalize name
            if (normalize o.SearchTerm).Contains normalizedName 
            then "match" 
            else "no match"
        |> Some)
        |> Categorical

let numberOfAttributes =
    "Number of attributes",
    (fun (o:Observation) -> 
        let count = o.Product.Attributes.Count
        if count = 0 then "none"
        elif count < 10 then "low"
        elif count < 20 then "med"
        else "hi"
        |> Some)
        |> Categorical
    
let descriptionMatch =
    "description match",
    (fun (o:Observation) -> 
        let desc = o.Product.Description |> wordTokenizer
        let query = o.SearchTerm |> wordTokenizer
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

let features = 
    [
        wordsInQuery
        brandMatch
        numberOfAttributes
        descriptionMatch
    ]

(* 
Experimentations 
*)


let learner sample =
    let tree = basicTree sample (labels,features) { DefaultSettings with Holdout = 0.2 }
    (tree.Classifier >> numerize)

(*
Validation
*)

let test = evaluate 5 learner

printfn "RMSE: %.3f" (test.RMSE)

//createSubmission learner
//
//printfn "Preparing predictions..."
