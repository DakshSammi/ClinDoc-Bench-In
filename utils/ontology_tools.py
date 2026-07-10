# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import logging
from typing import List, Dict, Any

class OntologyTools:
    def __init__(self, bioportal_key: str, bioportal_url: str, aberowl_url: str):
        self.bioportal_key = bioportal_key
        self.bioportal_url = bioportal_url
        self.aberowl_url = aberowl_url
        self.logger = logging.getLogger("OntologyTools")

    def search_bioportal(self, query: str, ontologies: List[str] = ["SNOMEDCT", "RXNORM"]) -> List[Dict[str, Any]]:
        self.logger.info(f"Searching BioPortal for: {query}")
        params = {
            "q": query,
            "ontologies": ",".join(ontologies),
            "apikey": self.bioportal_key,
            "pagesize": 5
        }
        try:
            response = requests.get(f"{self.bioportal_url}/search", params=params)
            response.raise_for_status()
            results = response.json().get("collection", [])
            return [{"id": r["@id"], "label": r["prefLabel"], "ontology": r["links"]["ontology"]} for r in results]
        except Exception as e:
            self.logger.error(f"BioPortal Error: {str(e)}")
            return []

    def search_aberowl(self, query: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Searching AberOWL for: {query}")
        try:
            response = requests.get(f"{self.aberowl_url}{query}")
            response.raise_for_status()
            results = response.json() # AberOWL format depends on the specific endpoint
            # Simplified mapping for demonstration
            return [{"id": r.get("iri"), "label": r.get("label"), "ontology": "AberOWL"} for r in results[:5]]
        except Exception as e:
            self.logger.error(f"AberOWL Error: {str(e)}")
            return []

    def get_candidates(self, query: str) -> List[Dict[str, Any]]:
        # Combine results from multiple sources
        candidates = self.search_bioportal(query)
        candidates.extend(self.search_aberowl(query))
        return candidates
