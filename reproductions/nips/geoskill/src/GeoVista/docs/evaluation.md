
## Inference & Evaluation Pipeline

### Nuanced Evaluation

To perform nuanced evaluation on GeoBench to obtain the haversine distance with the groud truth, you need to:

-  Set up Google Cloud API key

1. You need to have [Google Cloud](https://console.cloud.google.com/) account and set up your project.
2. Enable the [Geocoding API](https://console.cloud.google.com/marketplace/product/google/geocoding-backend.googleapis.com) for your project.
3. Create API [credentials](https://console.cloud.google.com/apis/credentials) and obtain your API key.

Finally, update the `GOOGLE_MAPS_API_KEY` variable of the `.env` file with your API key.

- Run the geocoding test script to verify the API key is working:

```bash
python3 eval/utils_geocode.py
```

If the test is successful, you should see an output like:

```text
[extract_pred_address] attempt=0 sending messages
[extract_pred_address] raw response: '{"address":"Einkaufszentrum Rahlstedt-Ost, Schöneberger Straße, 22149 Hamburg, Germany"}'
[extract_pred_address] parsed address: Einkaufszentrum Rahlstedt-Ost, Schöneberger Straße, 22149 Hamburg, Germany
[geocode] trying Google Geocoding: https://maps.googleapis.com/maps/api/geocode/json?address=Einkaufszentrum+Rahlstedt-Ost%2C+Sch%C3%B6neberger+Stra%C3%9Fe%2C+22149+Hamburg%2C+Germany&key=****
[geocode] Google status=OK, results=1
[geocode] Google success lat=53.5868597, lng=10.1533124
[geocode] trying Google Geocoding: https://maps.googleapis.com/maps/api/geocode/json?address=Tencent+beijing+Office%2C+Beijing%2C+China&key=****
[geocode] Google status=OK, results=1
[geocode] Google success lat=39.904211, lng=116.407395
```

- Finally, run the nuanced evaluation script and remove the `--no_eval_accurate_dist` flag:

```bash
MODEL_NAME=geovista-rl-12k-7b
BENCHMARK=geobench
EVALUATION_RESULT=".temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.jsonl"

python3 eval/eval_infer_geolocation.py \
  --pred_jsonl <The inference file path> \
  --out_jsonl ${EVALUATION_RESULT}\
  --dataset_dir .temp/datasets/${BENCHMARK} \
  --num_samples 1500 \
  --model_verifier \
  --timeout 120 --debug | tee .temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.log 2>&1
```

When the evaluation is done, you will find the haversine distance results in the `evaluation.jsonl` file under the specified output directory, also you will have a result csv file named `eval_summary.csv` containing the summarized evaluation metrics including the level-wise metrics and the haversine distance statistics.
