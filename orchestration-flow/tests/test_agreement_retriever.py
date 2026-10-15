from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from otl.agents.agreement_retriever import (
    _AgreementRanking,
    _RankedCandidate,
    _detected_ground_queries,
    _select_ranked,
    _validated_labels,
    agreement_retriever_node,
    auditor_query,
    shell_findings_query,
)
from otl.agents.drafter import _agreement_prompt_block
from otl.agreements.corpus import (
    AgreementCorpusManifest,
    AgreementDocumentSpec,
    AssetSpec,
    canonical_asset,
    documents_for_asset,
    load_manifest,
)
from otl.agreements.cache import sync_document_metadata
from otl.agreements.chunk import _paragraph_units, chunk_document
from otl.agreements.store import AgreementIndex, create_database
from otl.output_writer import assemble_report


class AgreementCorpusTests(unittest.TestCase):
    def test_great_white_party_variants_resolve_to_their_explicit_scopes(self) -> None:
        manifest = load_manifest()
        bp = canonical_asset("Great White (BP Only)", manifest)
        chevron = canonical_asset("GreatWhite - Chevron only", manifest)

        self.assertEqual(bp, "great_white_bp_only")
        self.assertEqual(chevron, "great_white_chevron_only")
        self.assertEqual(
            {doc.document_id for doc in documents_for_asset(bp, manifest)},
            {"great_white_uoa", "perdido_gw_attestation", "great_white_joa"},
        )
        self.assertEqual(
            {doc.document_id for doc in documents_for_asset(chevron, manifest)},
            {
                "great_white_uoa", "perdido_coua", "chevron_2019_settlement",
                "perdido_gw_attestation", "rig_usage_agreement", "great_white_joa",
            },
        )

    def test_mars_has_its_own_bp_otl_track(self) -> None:
        manifest = load_manifest()
        ids = {doc.document_id for doc in documents_for_asset("Mars et al", manifest)}

        self.assertTrue({
            "mars_1992_uoa_amendment", "bp_2018_otl_settlement", "bp_2019_rigs_settlement",
        }.issubset(ids))

    def test_ursa_and_europa_resolve_only_to_the_supported_bp_settlements(self) -> None:
        manifest = load_manifest()
        expected = {"bp_2018_otl_settlement", "bp_2019_rigs_settlement"}

        self.assertTrue(expected.issubset(
            {doc.document_id for doc in documents_for_asset("Ursa", manifest)}
        ))
        self.assertTrue(expected.issubset(
            {doc.document_id for doc in documents_for_asset("Europa", manifest)}
        ))

    def test_new_assets_without_a_source_agreement_remain_unmapped(self) -> None:
        manifest = load_manifest()

        for asset in ("Appomattox", "Sparta", "Vito", "Whale"):
            self.assertEqual(documents_for_asset(asset, manifest), [], asset)

    def test_cached_document_metadata_updates_chunk_scopes_without_reocr(self) -> None:
        """Manifest metadata changes must reach every cached source chunk."""
        manifest = load_manifest()
        document = next(doc for doc in manifest.documents if doc.document_id == "bp_2018_otl_settlement")
        cache = {
            "document": {**document.model_dump(), "asset_scopes": ["mars"]},
            "chunks": [{"asset_scopes": ["mars"], "party_scope": [], "document_id": document.document_id}],
        }

        self.assertTrue(sync_document_metadata(cache, document))
        self.assertEqual(cache["document"]["asset_scopes"], ["mars", "ursa", "europa"])
        self.assertEqual(cache["chunks"][0]["asset_scopes"], ["mars", "ursa", "europa"])

    def test_cached_chunk_renames_legacy_instrument_fields(self) -> None:
        manifest = load_manifest()
        document = next(doc for doc in manifest.documents if doc.document_id == "bp_2018_otl_settlement")
        cache = {
            "document": {"instrument_name": "legacy", "instrument_type": "legacy"},
            "chunks": [{
                "document_id": document.document_id,
                "instrument_name": "legacy",
                "instrument_type": "legacy",
            }],
        }

        self.assertTrue(sync_document_metadata(cache, document))
        chunk = cache["chunks"][0]
        self.assertEqual(chunk["document_name"], document.document_name)
        self.assertEqual(chunk["document_type"], document.document_type)
        self.assertNotIn("instrument_name", chunk)
        self.assertNotIn("instrument_type", chunk)

    def test_cached_chunk_metadata_is_repaired_even_if_document_metadata_is_current(self) -> None:
        manifest = load_manifest()
        document = next(doc for doc in manifest.documents if doc.document_id == "bp_2018_otl_settlement")
        cache = {
            "document": document.model_dump(),
            "chunks": [{"asset_scopes": ["mars"], "party_scope": [], "document_id": document.document_id}],
        }

        self.assertTrue(sync_document_metadata(cache, document))
        self.assertEqual(cache["chunks"][0]["asset_scopes"], ["mars", "ursa", "europa"])


class AgreementQueryTests(unittest.TestCase):
    def test_original_auditor_text_is_primary(self) -> None:
        state = {
            "auditor_narrative": "ORIGINAL auditor exception about affiliate labor.",
            "intake": {"nl_summary": "MODEL FINDINGS"},
            "rebuttal_current_posture": "MODEL FINDINGS",
        }

        self.assertEqual(
            auditor_query(state), "ORIGINAL auditor exception about affiliate labor."
        )

    def test_auditor_query_includes_latest_reply_without_dropping_original_baseline(self) -> None:
        state = {
            "auditor_narrative": "ORIGINAL request for a cost certificate.",
            "latest_auditor_reply": {"text": "The certificate remains outstanding for 2023."},
            "rebuttal_current_posture": "MODEL FINDINGS",
        }

        query = auditor_query(state)

        self.assertIn("ORIGINAL request for a cost certificate.", query)
        self.assertIn("The certificate remains outstanding for 2023.", query)

    def test_shell_query_adds_multiple_legal_routes_without_people_names(self) -> None:
        state = {
            "jade_analyzer": {
                "people_found_in_ledger": ["Person, One", "Person, Two", "Person, Three"],
                "major_construction_charge": False,
            },
            "job_title": {
                "technical_job_title": True,
                "identified_job_data": {
                    "BUSINESS ADMINISTRATION SUPPORT JG5": "Administrative and secretarial support.",
                    "TURNAROUND PLANNER/SCHEDULER JG5": "Plans and executes turnaround events.",
                },
            },
            "matching": {
                "identified_cfr_category": "Regulatory Activity",
                "cfr_rationale": {
                    "WELLS ENGINEER": "Drilling work requiring APD, casing, and BOP compliance.",
                    "PROJECT STUDIES": "Performs feasibility and project studies.",
                    "BUSINESS ADMINISTRATION SUPPORT JG5": (
                        "Excluded administration function recovered through overhead."
                    ),
                },
                "nl_summary": "The population is mixed and includes regulatory and feasibility routes requiring distinct agreement support.",
            },
        }

        with patch("otl.agents._common._max_people", 2):
            query = shell_findings_query(state)

        self.assertIn("regulatory activity", query.lower())
        self.assertIn("feasibility project studies", query.lower())
        self.assertIn("planning maintenance turnaround", query.lower())
        self.assertIn("technical employee exclusion", query.lower())
        self.assertIn("2 of 3 named people were examined", query)
        self.assertNotIn("Person, One", query)

    def test_shell_query_adds_mcoh_timewriting_threshold_without_people_names(self) -> None:
        state = {
            "jade_analyzer": {
                "major_construction_charge": True,
                "mcoh_monthly_facts": [
                    {"person": "Sensitive Name", "submission_month": "2023-01",
                     "hours": 6.0, "threshold_hours": 8.0, "below_threshold": True},
                    {"person": "Another Name", "submission_month": "2023-02",
                     "hours": 12.0, "threshold_hours": 8.0, "below_threshold": False},
                ],
            }
        }
        query = shell_findings_query(state)
        self.assertIn("major construction timewriting threshold", query.lower())
        self.assertIn("submitted-hour range 6-12", query)
        self.assertNotIn("Sensitive Name", query)
        self.assertNotIn("Another Name", query)

    def test_shell_query_uses_offsite_technical_labor_and_cfr_context(self) -> None:
        state = {
            "job_title": {
                "role_status_by_person": {"Sensitive Name": "technical"},
                "positions_by_person": {"Sensitive Name": ["Staff Wells Engineer"]},
                "identified_job_data": {"Wells Engineer": "Designs and executes well operations."},
            },
            "matching": {
                "cfr_mapping_status_by_person": {"Sensitive Name": "matched"},
                "cfr_primary_by_person": {"Sensitive Name": ["30 CFR 250 Subpart D"]},
                "subpart_labels_by_person": {"Sensitive Name": ["Drilling Operations"]},
            },
            "location": {"location_status": {"Sensitive Name": "offsite"}},
        }
        query = shell_findings_query(state)
        self.assertIn("offsite technical labor", query.lower())
        self.assertIn("30 CFR 250 Subpart D", query)
        self.assertNotIn("Sensitive Name", query)

    def test_detected_ground_queries_keep_distinct_auditor_and_shell_routes(self) -> None:
        grounds = dict(_detected_ground_queries(
            "Affiliate labor may instead qualify as regulatory activity, feasibility studies, "
            "or planners and turnaround work; administrative roles are excluded overhead."
        ))

        self.assertIn("regulatory_activity", grounds)
        self.assertIn("feasibility_project_studies", grounds)
        self.assertIn("planning_maintenance_turnaround", grounds)
        self.assertIn("technical_employee_exclusion", grounds)
        self.assertIn("affiliate_labor", grounds)


class AgreementSelectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(number: int, track: str, perspective: str) -> dict:
        return {
            "chunk_id": f"c{number}",
            "document_id": f"d{number}",
            "document_name": f"Document {number}",
            "legal_track": track,
            "verbatim_text": f"Unique governing clause text number {number}.",
            "perspectives": [perspective],
        }

    def test_top_five_uses_validated_labels_and_reranker_order(self) -> None:
        candidates = {
            f"c{i}": self._candidate(
                i, f"track-{i}", "auditor_theory" if i == 1 else "shell_findings"
            )
            for i in range(1, 7)
        }
        ranking = _AgreementRanking(ranked_candidates=[
            _RankedCandidate(
                chunk_id=f"c{i}", relevance="governing", perspectives=candidates[f"c{i}"]["perspectives"],
                relationship="supports", selection_reason="relevant",
            )
            for i in range(1, 7)
        ])

        selected = _select_ranked(_validated_labels(ranking, candidates), 5)

        self.assertEqual(len(selected), 5)
        self.assertIn("auditor_theory", {p for item, _ in selected for p in item.perspectives})
        self.assertIn("shell_findings", {p for item, _ in selected for p in item.perspectives})
        self.assertEqual(len({candidate["legal_track"] for _, candidate in selected}), 5)


class AgreementChunkingTests(unittest.TestCase):
    def test_numbered_clauses_split_without_blank_ocr_paragraphs(self) -> None:
        units = _paragraph_units(
            "ATTACHMENT 1\nAGREED OTL DEFINITION AND BILLING STANDARDS\n"
            "1. Definition of Offsite Technical Labor\nDefinition body line one.\nBody line two.\n"
            "2. Definition of Technical Employees\nSecond definition body."
        )

        self.assertIn("1. Definition of Offsite Technical Labor", units)
        self.assertIn("2. Definition of Technical Employees", units)
        self.assertTrue(any("Definition body line one" in unit for unit in units))

    def test_major_heading_starts_a_clean_chunk_without_prior_overlap(self) -> None:
        document = AgreementDocumentSpec(
            document_id="doc", relative_path="unused.pdf", document_name="Agreement",
            document_type="Settlement Agreement", legal_track="settlement",
            asset_scopes=["great_white"], party_scope=["Shell", "BP"],
        )
        pages = [{
            "page": 1,
            "printed_page_label": "1",
            "confidence": 0.99,
            "method": "ocr",
            "text": ("signature material " * 250) + "\n\nATTACHMENT 1\n\n"
                    + ("Technical Employees are chargeable for qualifying work. " * 80),
        }]

        chunks = chunk_document(document, pages)
        attachment = next(chunk for chunk in chunks if "ATTACHMENT 1" in chunk["verbatim_text"])

        self.assertTrue(attachment["verbatim_text"].startswith("ATTACHMENT 1"))
        self.assertNotIn("signature material", attachment["verbatim_text"])
        self.assertEqual(attachment["extraction_methods"], ["ocr"])

    def test_attachment_boundary_flushes_even_a_short_signature_preamble(self) -> None:
        document = AgreementDocumentSpec(
            document_id="doc", relative_path="unused.pdf", document_name="Agreement",
            document_type="Settlement Agreement", legal_track="settlement",
            asset_scopes=["great_white"], party_scope=["Shell", "BP"],
        )
        chunks = chunk_document(document, [{
            "page": 1,
            "printed_page_label": "1",
            "confidence": 0.99,
            "method": "ocr",
            "text": "By: Name: Title:\n\nATTACHMENT 1\n\n1. Definition of Technical Employees\n"
                    + ("Technical employees perform qualifying engineering work. " * 20),
        }])

        attachment = next(chunk for chunk in chunks if "ATTACHMENT 1" in chunk["verbatim_text"])
        self.assertTrue(attachment["verbatim_text"].startswith("ATTACHMENT 1"))
        self.assertNotIn("By: Name: Title", attachment["verbatim_text"])

    def test_repeated_source_text_never_emits_duplicate_chunk_ids(self) -> None:
        document = AgreementDocumentSpec(
            document_id="doc", relative_path="unused.pdf", document_name="Agreement",
            document_type="Settlement Agreement", legal_track="settlement",
            asset_scopes=["great_white"], party_scope=["Shell", "BP"],
        )
        repeated = "ATTACHMENT 1\n\n" + ("Technical employee agreement language. " * 80)
        chunks = chunk_document(document, [{
            "page": 1, "printed_page_label": "1", "confidence": 0.99, "method": "ocr",
            "text": repeated + "\n\n" + repeated,
        }])

        ids = [chunk["chunk_id"] for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))


class AgreementNodeTests(unittest.TestCase):
    def _manifest_and_index(self, directory: str):
        documents = [
            AgreementDocumentSpec(
                document_id="shared", relative_path="unused.pdf", document_name="Shared UOA",
                document_type="UOA", legal_track="shared_uoa", asset_scopes=["great_white"],
                party_scope=["Shell", "BP", "Chevron"],
            ),
            AgreementDocumentSpec(
                document_id="bp", relative_path="unused2.pdf", document_name="BP Settlement",
                document_type="Settlement Agreement", legal_track="bp_settlement",
                asset_scopes=["great_white"], party_scope=["Shell", "BP"],
            ),
        ]
        manifest = AgreementCorpusManifest(
            assets={"great_white": AssetSpec(aliases=["Great White"])},
            documents=documents,
            fingerprint="fixture-fingerprint",
        )
        caches = []
        for index, doc in enumerate(documents, 1):
            text = (
                "Affiliate technical labor is chargeable when the employee performs regulatory "
                f"work for Joint Operations. Exact source clause {index}."
            )
            caches.append({
                "document": doc.model_dump(),
                "source_path": f"/{doc.document_id}.pdf",
                "source_sha256": f"sha-{doc.document_id}",
                "pages": [{"page": index, "text": text}],
                "chunks": [{
                    "chunk_id": f"chunk-{doc.document_id}",
                    "document_id": doc.document_id,
                    "document_name": doc.document_name,
                    "document_type": doc.document_type,
                    "legal_track": doc.legal_track,
                    "asset_scopes": doc.asset_scopes,
                    "party_scope": doc.party_scope,
                    "heading_path": "Technical Labor",
                    "pdf_page_start": index,
                    "pdf_page_end": index,
                    "printed_page_labels": [str(index)],
                    "extraction_methods": ["ocr"],
                    "verbatim_text": text,
                    "ocr_confidence": 0.99,
                    "quote_safe": True,
                    "content_sha256": f"content-{index}",
                    "token_count": 25,
                }],
            })
        database = Path(directory) / "index.sqlite3"
        create_database(database, caches, manifest)
        return manifest, AgreementIndex(database)

    def test_llm_ranks_ids_and_node_rehydrates_exact_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, index = self._manifest_and_index(directory)
            ranking = _AgreementRanking(ranked_candidates=[
                _RankedCandidate(
                    chunk_id="chunk-bp", relevance="direct",
                    perspectives=["auditor_theory", "shell_findings"],
                    relationship="supports", selection_reason="BP-specific rule",
                ),
                _RankedCandidate(
                    chunk_id="chunk-shared", relevance="supporting",
                    perspectives=["auditor_theory"], relationship="context",
                    selection_reason="shared agreement",
                ),
            ])
            state = {
                "case": {"case_id": "case-1", "asset": "Great White", "year": "2025"},
                "auditor_narrative": "Affiliate technical labor regulatory chargeability",
                "matching": {"nl_summary": "Regulatory activity supports chargeability under the identified authority."},
            }
            with patch("otl.agents.agreement_retriever.load_manifest", return_value=manifest), patch(
                "otl.agents.agreement_retriever.AgreementIndex", return_value=index
            ), patch(
                "otl.agents.agreement_retriever.llm.structured", return_value=ranking
            ) as rerank:
                result = agreement_retriever_node(state)

        sub = result["agreement_retriever"]
        self.assertEqual(sub.status, "matched")
        self.assertEqual(len(sub.selected_chunks), 2)
        self.assertEqual(
            sub.selected_chunks[0].verbatim_text,
            "Affiliate technical labor is chargeable when the employee performs regulatory work "
            "for Joint Operations. Exact source clause 2.",
        )
        self.assertEqual(rerank.call_count, 1)
        self.assertNotIn("verbatim_text", ranking.model_dump())
        self.assertEqual(sub.agreement_chargeability, "chargeable")
        self.assertEqual(
            [item["chunk_id"] for item in sub.agreement_chargeability_citations],
            ["chunk-bp", "chunk-shared"],
        )

    def test_report_surfaces_agreement_output(self) -> None:
        report = assemble_report(
            {
                "case": {"case_id": "case-2", "asset": "Mars"},
                "agreement_retriever": {"status": "matched", "selected_chunks": []},
                "drafter": {"disposition": "deny"},
            },
            run_id="case-2", elapsed_minutes=0.1, token_usage=None, trace_events=[],
        ).model_dump()

        self.assertEqual(report["agreement_output"]["status"], "matched")
        self.assertEqual(report["stages"]["agreement_retriever"]["status"], "matched")


class DrafterAgreementPromptTests(unittest.TestCase):
    def test_selected_chunk_reaches_drafter_without_truncation(self) -> None:
        source = "Beginning " + ("exact agreement language " * 180) + " End marker."
        block = _agreement_prompt_block({
            "agreement_retriever": {
                "status": "matched",
                "selected_chunks": [{
                    "chunk_id": "chunk-1",
                    "document_name": "Test Agreement",
                    "legal_track": "bp_settlement",
                    "heading_path": "Section 3",
                    "pdf_page_start": 10,
                    "pdf_page_end": 11,
                    "printed_page_labels": ["8", "9"],
                    "extraction_methods": ["ocr"],
                    "party_scope": ["Shell", "BP"],
                    "quote_safe": True,
                    "content_sha256": "abc",
                    "verbatim_text": source,
                }],
            }
        })

        self.assertIn(source, block)
        self.assertIn("[/AGREEMENT_CHUNK]", block)
        self.assertNotIn("additional item", block)


if __name__ == "__main__":
    unittest.main()
