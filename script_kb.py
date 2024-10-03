import os
import json
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class ScriptKB:

    def __init__(self,file_path):

        with open(file_path,'r') as infile:
            
            self.story = json.load(infile)

    def get_title(self):

        return self.story["title"]
        
    def get_characters(self):

        return list(self.story["characters"].keys())

    def get_character_utterance_count(self,character_name):

        return self.story["characters"][character_name]["utterances"]

    def get_character_profile(self,character_name):

        return self.story["characters"][character_name]["persona"]["zero-shot"]

    def get_character_utterances(self,character_name):

        utterances = []

        for item in self.story["content"]:

            if(item["content_type"]=="utterance" and item["character"]==character_name):
                utterances.append(item)

        return utterances

    def get_kb_ending_at(self,timestamp):

        event_list = []

        for item in self.story["content"]:

            if(item["timestep"]>timestamp):
                return event_list
            event_list.append(item)

        return event_list

    def get_kb_between(self,start_timestamp=None,end_timestamp=None):

        event_list = []

        for item in self.story["content"]:

            if(end_timestamp is not None and item["timestep"]==end_timestamp):
                return event_list
            if(start_timestamp is None):
                event_list.append(item)
            elif(item["timestep"]>=start_timestamp):
                event_list.append(item)

        return event_list

    def get_scene_settings(self):
        
        event_list = []

        for item in self.story["content"]:

            if(item["content_type"]=="setting"):
                event_list.append(item)

        return event_list

    def num_events(self):

        return len(self.story["content"])

    def num_series_items(self):
        
        return len(self.story["files"])

    def num_scenes(self):

        event_list = []

        for item in self.story["content"]:

            if(item["content_type"]=="setting"):
                event_list.append(item)

        return len(event_list)

    def scene_times(self):

        timestamps = []

        for item in self.story["content"]:

            if(item["content_type"]=="setting"):
                timestamps.append(item["timestep"]) 

        return timestamps

    def get_scene_content(self,kb):

        if(kb["content_type"]=="utterance"):
            scene_description = kb["character"] + ": "+ kb["text"]
        else:
            scene_description = kb["text"]
    
        return scene_description

class Script_Vector_KB:

    def __init__(self,file_path):

        with open(file_path,'r') as infile:
            
            self.story = json.load(infile)
        memory_set = set()
        self.memory_contents = []
        for x in self.story["content"]:
            if(x["text"] not in memory_set):
                memory_set.add(x["text"])
                self.memory_contents.append(x)
                
        self.retreival_model = SentenceTransformer('all-mpnet-base-v2')
        self.index = None
        self.build_memories()

    def build_memories(self):

        documents = {}

        for xi,x in enumerate(self.memory_contents):
            documents[xi]=x["text"]
        embeddings, doc_ids = self.encode_memories(documents)
        self.build_index(embeddings)
        

    def encode_memories(self, documents):
        # expects documents to be a dict with doc_id and doc_text
        doc_ids = list(documents.keys())
        doc_texts = list(documents.values())

        # SentenceTransformer supports batch processing
        embeddings = self.retreival_model.encode(doc_texts)

        return embeddings, np.array(doc_ids)

    def build_index(self, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def get_similar_memories(self, query, k=5):
        
        query_embedding = self.encode_memories({0: query})[0]
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)
        
        memories = []

        for mi in I[0,:].tolist():
            if(mi!=-1):
                memories.append(self.get_scene_content(self.memory_contents[mi])+"\n")
            
        return memories

    def get_scene_content(self,kb):

        if(kb["content_type"]=="utterance"):
            scene_description = kb["character"] + ": "+ kb["text"]
        else:
            scene_description = kb["text"]
    
        return scene_description