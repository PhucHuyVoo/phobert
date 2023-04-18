# PhoBERT Sentiment Analysis
## Step to Reproduce
- Step 1: Clone the repository
- Step 2: Make sure the clean csv is in the repository. You can use your custom file
- Step 3: Run `docker compose up -d`
- Step 4: Check container id using `docker ps`. Double check the container is up 
- Step 5: Run `docker exec -it {container_id} /bin/bash` to access to container bash
- Step 6: Run `python main.py {csv_file_name}` to do sentiment analysis on the current file
- Step 7: The results will save on the `/src` folder on Docker Container or `/data` folder on local repository 
