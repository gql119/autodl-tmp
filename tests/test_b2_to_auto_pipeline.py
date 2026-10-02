"""Infrastructure unit checks; fixtures retained, no GPU/data/model invocation."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import os

spec=importlib.util.spec_from_file_location('auto_pipeline',Path(__file__).resolve().parents[1]/'scripts/run_b2_to_auto_pipeline.py')
pipeline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.original=pipeline.ROOT
        self.root=self.original/'tests'/f'pipeline_cpu_fixture_{time.time_ns()}'
        self.root.mkdir()
        pipeline.ROOT=self.root
        self.config=self.root/'config.yaml'; self.config.write_text('test: infrastructure_only\n')
        paths=['ue_framework/methods/b2_semantic.py','ue_framework/methods/semantic_coordinates.py',
               'ue_framework/methods/orth_reference_design.py','ue_framework/methods/b2_adaptive.py',
               'ue_framework/methods/adaptive_learner.py','scripts/train_noise_once.py',
               'scripts/materialize_noise_once.py','scripts/run_isolated_victim.py']
        hashes={}
        for rel in paths:
            p=self.root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('# fixture only\n')
            hashes[rel]=pipeline.sha256(p)
        self.receipt={'sanity_gate':'passed','scientific_review':'scientifically_correct',
                      'config_sha256':pipeline.sha256(self.config),'file_sha256':hashes}
        self.admission=self.root/'admission.json'
        self.admission.write_text(json.dumps(self.receipt))

    def tearDown(self):
        pipeline.ROOT=self.original
        # Deliberately preserve all fixture files; no recursive cleanup.

    def test_gate_rejection_precedes_any_subprocess(self):
        self.receipt['sanity_gate']='failed'; self.admission.write_text(json.dumps(self.receipt))
        with self.assertRaises(RuntimeError): pipeline.validate_admission(self.config,self.admission)

    def test_changed_source_rejected(self):
        pipeline.validate_admission(self.config,self.admission)
        (self.root/'scripts/train_noise_once.py').write_text('# modified fixture\n')
        with self.assertRaises(RuntimeError): pipeline.validate_admission(self.config,self.admission)

    def test_failed_stage_records_exit_and_raises(self):
        status=self.root/'unit_stage.json'
        with self.assertRaises(RuntimeError):
            pipeline.run_stage('noise',[sys.executable,'-c','raise SystemExit(7)'],
                               status,{},dict(os.environ))
        self.assertEqual((self.root/'unit_stage.noise.exitcode').read_text().strip(),'7')
        self.assertEqual(json.loads(status.read_text())['stage'],'noise')


if __name__=='__main__':unittest.main()
