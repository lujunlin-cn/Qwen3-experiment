# Gravity DPO-B SFT C evidence index

Final status: `no_preference_signal`. GPU rollout completed; DPO training, checkpoint reload, Tune and Locked Final were gated off. No effect comparison exists.

- `final_quality_summary_v12.json`: authoritative full-pool quality statistics.
- `final_rollouts_audited_v12.jsonl`: all312 original responses plus independent verifier outcomes.
- `final_preference_pairs_v12.jsonl`: empty, correct outcome; no synthetic Chosen/Rejected.
- `rollout8/SFT_C_raw.jsonl`:208 original GPU responses, Token IDs and behavior logprobs.
- `rollout4/SFT_C_raw.jsonl`:104 supplemental responses, no repeats of original candidates.
- `identity_audit.json`, `data_audit.json`, `train_panel.jsonl`: model and real/synthetic Train lineage.
- `registration.json`: original source/config and48/100 validation panel registration.
- `verification_amendment.json`: pre-preference CPU-adversarial strengthening v1.1.
- `offline_acceptance_amendment_v12.json`: final CPU-only correctness/format separation and stricter optional fields; no new generation.
- `cpu_tests.log`:60 tests before initial GPU execution.
- `cpu_verifier_repair_gate.log`:64 tests before the supplemental GPU stage.
- `cpu_final_tests.log`:69 final CPU tests passed in4.455s.
- `official_correct_but_unverified_cases.jsonl`: case evidence, no model self-grading.
- `final_publication_manifest.json`: final sources, report and archived artifact hashes.
- `final_integrity_check.json`: actual hash, provenance protection and EOS accounting check.
- `final_status.json`, `automatic_final_report_v11.md`: unchanged v1.1 automatic outcome; use v1.2 quality summary for final diagnostic definitions.
- `attempts/`: preserved old statuses/audits and EOS-summary correction; excluded from final statistics.
- `driver.log`, `continuation_driver*.log`, `failures.jsonl`, `stage_hooks.jsonl`: complete failed/recovered completion-chain evidence.

The pinned model has two valid EOS tokens (151645 and151643). All are included in the287099 response tokens. Raw files are UTF-8 JSONL and unmodified. No API key or Authorization header was archived. No DPO checkpoint was created.

The formal report is `docs/DPO_GRAVITY_SFT_C_FINAL_REPORT.md`. Do not run the registered ID again or overwrite any file. A new experiment requires a new run and protocol.
