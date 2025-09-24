## Setting up Data processing. 
### Usage for Kafka setup and generating csv
- Select virtual environment or use default environment
```
python -m venv .venv

#activate environment

source .venv/bin/activate
```

- Install required dependencies:

```
pip install -e .

```

- Set environment variables

```
TUNNEL_IP=<kafka_IP>
TUNNEL_PORT=<kafka_port>
TUNNEL_USER=<kafka_username>
TUNNEL_PASSWORD=<kafka_password>
DATA_QUEUE_DIR=<output_dir>
DB_PATH=<db_dir>

```

- Run src/data_processing/process.py

```
python src/data_processing/process.py

```

### Usage for DB Writer (writing csv files to db)
- Manual (run db_writer)
```
python src/data_processing/db_writer.py
```

- For in-program usage, see lines 109-111
```
    writer = DBWriter()
    writer.process_files()
    writer.close()
```


### Usage for DB Reader (reading content of db)

- Manual (Run db_reader)
```
python src/data_processing/db_writer.py
```

- For in-program usage, see lines 51-54
```
    reader = DBReader()
    print(reader.get_watch_history())
    print(reader.get_ratings())
    reader.close()
```