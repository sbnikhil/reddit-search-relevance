import apache_beam as beam
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

class MarkUtility(beam.DoFn):
    def __init__(self, keywords):
        self.keywords = keywords

    def process(self, element):
        text = (element.get('body') or "").lower()
        if any(word in text for word in self.keywords):
            yield (f"{element['author']}||{element['subreddit']}", 1)

def run():
    project_id = config['database']['project_id']
    keywords = config['features']['utility_keywords']
    source = config['database']['source_table'].replace(':', '.')
    destination = f"{project_id}:{config['database']['dataset']}.user_utility_features"
    options = beam.options.pipeline_options.PipelineOptions(
        project=project_id,
        region='us-central1', 
        temp_location=f'gs://{project_id}-temp/tmp' 
    )

    with beam.Pipeline(options=options) as p:
        (
            p 
            | 'ReadFromBQ' >> beam.io.ReadFromBigQuery(
                query=f'SELECT author, subreddit, body FROM `{source}`', 
                use_standard_sql=True,
                project=project_id,
                method=beam.io.ReadFromBigQuery.Method.DIRECT_READ 
            )
            | 'ExtractUtility' >> beam.ParDo(MarkUtility(keywords))
            | 'SumHits' >> beam.CombinePerKey(sum)
            | 'FormatForBQ' >> beam.Map(lambda x: {
                'author': x[0].split('||')[0],
                'subreddit': x[0].split('||')[1],
                'utility_hit': x[1]
            })
            | 'WriteToBQ' >> beam.io.WriteToBigQuery(
                destination,
                schema='author:STRING, subreddit:STRING, utility_hit:INTEGER',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS
            )
        )

if __name__ == "__main__":
    print("Launching Utility Extraction Pipeline...")
    run()