﻿namespace HomeDepot

module Model =

    open System
    open System.IO
    open FSharp.Collections.ParallelSeq

    open FSharp.Data

    open HomeDepot.Utilities

    (*
    Prelude: core types
    *)

    type Attributes = Map<string,string>

    type Product = {
        UID:int
        Title:string
        Description:string
        Attributes:Attributes
        }

    type Observation = {
        ID:int
        SearchTerm:string
        Product:Product
        }

    type Relevance = float
    type Example = Relevance * Observation

    type Predictor = Observation -> Relevance

    type Learner = Example [] -> Predictor

    type Quality = {
        RMSE:float }

    (*
    Data loading
    *)

    let inline normalize (txt:string) = preprocess txt

    [<Literal>]
    let trainPath = @"../data/train.csv"
    [<Literal>]
    let testPath = @"../data/test.csv"
    [<Literal>]
    let attributesPath = @"../data/attributes.csv"
    [<Literal>]
    let productsPath =  @"..\data\product_descriptions.csv"

    [<Literal>]
    let submissionPath =  @"../data/"

    type Train = CsvProvider<trainPath,Schema=",,,,float">
    type Test = CsvProvider<testPath>
    type AllAttributes = CsvProvider<attributesPath>
    type AllProducts = CsvProvider<productsPath>

    let descriptions =

        printfn "Loading product descriptions"

        AllProducts.GetSample().Rows
        |> PSeq.ordered
        |> PSeq.map (fun row -> 
            row.Product_uid, 
            row.Product_description |> descriptionSentenceBreak |> normalize)
        |> Seq.timedSlim (Some 10000) Scale.ms
        |> dict

    let preprocessedAttributes =

        printfn "Pre-processing attributes"

        AllAttributes.GetSample().Rows
        |> PSeq.ordered
        |> PSeq.map (fun x ->
            x.Product_uid, 
            x.Name |> normalize, 
            x.Value |> normalize)
        |> Seq.timedSlim (Some 100000) Scale.ms
        |> Seq.toArray
        
    let attributes =

        printfn "Loading attributes"

        preprocessedAttributes
        |> PSeq.ordered
        |> PSeq.map (fun (id,name,value) -> name,value)
        |> Seq.groupBy fst
        |> Seq.map (fun (key,values) ->
            key,
            values |> Seq.map snd |> set)
        |> Map.ofSeq

    let productAttributes =

        printfn "Loading product attributes"

        preprocessedAttributes
        |> Seq.groupBy (fun (id,name,value) -> id)
        |> PSeq.ordered
        |> PSeq.map (fun (uid,rows) ->
            uid,
            rows
            |> Seq.map(fun (id,name,value) -> name,value)
            |> Map.ofSeq)
        |> dict

    let attributesFor (uid:int) =
        match productAttributes.TryGetValue uid with
        | true,values -> values
        | false,_     -> Map.empty

    let trainset =

        printfn "Loading train data"

        Train.GetSample().Rows
        |> Seq.map (fun row ->
            let description = descriptions.[row.Product_uid]
            let attributes = attributesFor row.Product_uid
            let product =
                {
                    UID = row.Product_uid
                    Title = row.Product_title |> normalize
                    Description = description
                    Attributes = attributes
                }
            // Fully constructed example
            row.Relevance,
            {
                ID = row.Id
                SearchTerm = row.Search_term |> normalize |> cleanMisspellings |> cleanSpaces
                Product = product
            })
        |> Seq.timedSlim (Some 10000) Scale.ms
        |> Seq.toArray

    let testset =

        printfn "Loading test data"

        Test.GetSample().Rows
        |> Seq.map (fun row ->
            let description = descriptions.[row.Product_uid]
            let attributes = attributesFor (row.Product_uid)
            let product =
                {
                    UID = row.Product_uid
                    Title = row.Product_title |> normalize
                    Description = description
                    Attributes = attributes
                }
            // Fully constructed observation
            {
                ID = row.Id
                SearchTerm = row.Search_term |> normalize |> cleanMisspellings |> cleanSpaces
                Product = product
            })
        |> Seq.timedSlim (Some 10000) Scale.ms
        |> Seq.toArray

    let rmse sample =
        sample
        |> Seq.averageBy (fun (act, exp) ->
            let delta = act - exp
            delta * delta)
        |> sqrt

    let evaluate (k:int) (learner:Learner) =

        let rng = Random(123456)
        let indexes =
            Array.init (trainset.Length) (fun _ -> rng.Next k)

        let evaluation =
            [ 0 .. (k - 1)]
            |> List.map (fun block ->

                printfn "Evaluating block %i" (block+1)

                let hold, used = 
                    indexes 
                    |> Array.mapi (fun i block -> i, block) 
                    |> Array.partition (fun (index,b) -> b = block)

                let holdSet = hold |> Array.map (fun (i,_) -> trainset.[i])
                let usedSet = used |> Array.map (fun (i,_) -> trainset.[i])

                printfn "  Learning"
                let predictor = learner usedSet

                printfn "  Evaluating"
                let quality =
                    holdSet
                    |> Seq.map (fun (actual, obs) ->
                        actual, predictor obs)
                    |> rmse
                block, { RMSE = quality } )

        evaluation
        |> List.iter (fun (block,quality) ->
            printfn "  %i: %.3f" block (quality.RMSE))

        let overall = evaluation |> Seq.averageBy (fun (_,x) -> x.RMSE)

        printfn "Overall: %.3f" overall
        { RMSE = overall }


    let createSubmission (learner:Learner) =

        printfn "Creating submission"

        printfn "Learning model"

        let predictor = learner trainset

        printfn "Generating predictions"

        let header = "id,relevance"
        let submission =
              [|
                  yield header
                  for case in testset ->
                      sprintf "%i,%.2f" (case.ID) (predictor case)
              |]

        printfn "Saving predictions"

        let filename = DateTime.Now.ToString("yyyy-MM-dd-HH-mm")

        let desktop =
            Environment.SpecialFolder.Desktop
            |> Environment.GetFolderPath

        let targetFolder = Path.Combine(desktop,"kaggle")

        if (not (Directory.Exists targetFolder))
        then
            Directory.CreateDirectory(targetFolder)
            |> ignore

        let filePath = Path.Combine(targetFolder,filename)

        File.WriteAllLines(filePath, submission)

        printfn "Predictions saved at %s" filePath
