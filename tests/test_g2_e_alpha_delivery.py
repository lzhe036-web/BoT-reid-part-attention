"""Synthetic evidence integration tests. No fixture is a measured experiment."""
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
import yaml
from PIL import Image
from config import cfg
from tests import test_recover_g2_global_local_experiment as fixture_support
from tests.test_g2_e_static_dynamic_alpha0p3_tau0p5 import model
from utils.config_serialization import serialize_cfg_node_yaml
from utils.experiment_recording import sha256_file
from utils.g2_e_alpha_delivery import read_json, read_rows, validate_raw, require_registered, archive_package
from utils.g2_e_checkpoint_identity import seal, validate
from tools import recover_g2_global_local_experiment as recovery
from tools.finish_g2_e_alpha_experiment import finish, module_for

ROOT = Path(__file__).resolve().parents[1]
TOKEN = '0p1' if (ROOT/'tools/finalize_g2_e_static_dynamic_alpha0p1_tau0p5_experiment.py').exists() else '0p5'
ALPHA = float(TOKEN.replace('p','.'))
STEM = 'g2_e_static_dynamic_alpha{}_tau0p5'.format(TOKEN)
CONFIG = ROOT/'configs'/('softmax_triplet_c2_l03_multi_granularity_dynamic_gating_'+STEM+'_autodl.yml')
BASE_CONFIG = ROOT/'configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e_static_dynamic_alpha0p3_tau0p5_autodl.yml'
analysis = importlib.import_module('tools.analyze_'+STEM)
profile = module_for('recover', ALPHA)
finalizer = module_for('finalize', ALPHA)
packager = module_for('package', ALPHA)
protocol = importlib.import_module('tools.verify_'+STEM+'_protocol')


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


class AlphaProtocolTest(unittest.TestCase):
    def test_full_resolved_config_only_alpha_and_output_change(self):
        report=protocol.verify(BASE_CONFIG,CONFIG)
        self.assertEqual(set(report['allowed_differences']), {'MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA','OUTPUT_DIR'})
        current=cfg.clone();current.merge_from_file(str(CONFIG))
        self.assertEqual(current.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA,.3)
        self.assertEqual(current.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA,ALPHA)
        data=yaml.safe_load(CONFIG.read_text())
        data['SOLVER']['BASE_LR']=.002
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.yml';p.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(protocol.ProtocolError,'Unexpected'):
                protocol.verify(BASE_CONFIG,p)

    def test_formula_initialization_gradients_and_same_seed_parameters(self):
        torch.set_num_threads(2)
        previous=None
        for alpha in (.1,.5):
            torch.manual_seed(42)
            network=model(True,alpha=alpha);network.eval();network.neck_feat='before'
            if previous is not None:
                for k,v in network.state_dict().items():self.assertTrue(torch.equal(v,previous[k]),k)
            previous={k:v.clone() for k,v in network.state_dict().items()}
            x=torch.randn(2,3,8,4)
            fmap=network.base(x);g=network.gap(fmap).view(2,-1);local=network.multi_granularity_part_head(fmap)
            result=network(x)
            expected=torch.cat((g,)+tuple((1+alpha)*z for z in local),dim=1)
            self.assertEqual(tuple(result.shape),(2,2816))
            self.assertTrue(torch.equal(result[:,:2048],g))
            self.assertTrue(torch.allclose(result,expected,atol=1e-6,rtol=1e-6))
            result.square().mean().backward()
            for parameter in (network.base.projection.weight,next(network.multi_granularity_part_head.parameters()),network.multi_granularity_dynamic_gate.controller.weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()),0)


class DeliveryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.saved_globals={k:v for k,v in vars(recovery).items() if k.isupper()}
        self.fixture=fixture_support.G2RecoveryTest();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.addCleanup(lambda:vars(recovery).update(self.saved_globals))
        f=self.fixture
        self.configuration=cfg.clone();self.configuration.merge_from_file(str(CONFIG))
        self.configuration.OUTPUT_DIR=str(f.output)
        self.dataset=f.root/'dataset';self.dataset.mkdir()
        self.configuration.DATASETS.ROOT_DIR=str(self.dataset)
        source=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
        source['OUTPUT_DIR']=str(f.output)
        source['DATASETS']['ROOT_DIR']=str(self.dataset)
        f.config.write_text(yaml.safe_dump(source),encoding='utf-8')
        resolved=f.output/'config_resolved.yml'
        resolved.write_text(serialize_cfg_node_yaml(self.configuration),encoding='utf-8')
        provenance=read_json(f.output/'reproducibility.json')
        provenance['code']['branch']=profile.EXPECTED_BRANCH
        provenance['configuration']['source_file_sha256']=sha256_file(f.config)
        provenance['configuration']['resolved_file_sha256']=sha256_file(resolved)
        write_json(f.output/'reproducibility.json',provenance)
        gate_path=f.output/'dynamic_gating_epoch_stats.jsonl'
        gate_records=[json.loads(line) for line in gate_path.read_text().splitlines()]
        for row in gate_records:row['gating_temperature']=.5
        gate_path.write_text(''.join(json.dumps(r)+'\n' for r in gate_records))
        old_result=read_json(f.output/'g2_formal_result.json')
        self.result=dict(old_result,experiment_id=profile.EXPERIMENT_ID,branch=profile.EXPECTED_BRANCH,
                         static_dynamic_alpha=ALPHA,gating_temperature=.5,fusion_mode='static_concat_plus_gated_residual',
                         fusion_formula='concat(g,(1+alpha*w2)z2,(1+alpha*w4)z4,(1+alpha*w6)z6)',
                         gating_input_dim=2816,controller='Linear(2816,3)',controller_parameter_count=8451,descriptor_dim=2816)
        self.result['selected_epoch_gate_statistics']=gate_records[79]
        self.result['evidence']['config_sha256']=sha256_file(f.config)
        self.result['evidence']['epoch_gate_statistics_sha256']=sha256_file(gate_path)
        self.checkpoint=Path(self.result['selected_checkpoint']['path'])
        (f.output/'log.txt').write_text(''.join('EPOCH_EVIDENCE epoch={} global_iteration={} epoch_length=186\n'.format(e,e*186) for e in (40,80,120)))
        seal(f.output)
        from tools.analyze_dynamic_gating import _stable_key
        samples=[];self.mapping={}
        for index in range(6):
            image=self.dataset/('sample_{}.jpg'.format(index));Image.new('RGB',(24,48),(index*30,90,150)).save(image)
            split='query' if index<2 else 'gallery'
            key=_stable_key(split,str(image),index,0,str(self.dataset))
            p=(.1,.7,.2) if index<5 else (.1,.2,.7)
            row=dict(stable_sample_key=key,dataset_split=split,pid=str(index),camid='0',dominant_k='4' if index<5 else '6',checkpoint_sha256=self.result['selected_checkpoint']['sha256'])
            for k,v in zip((2,4,6),p):row['p'+str(k)]=v;row['w'+str(k)]=3*v
            samples.append(row);self.mapping[key]=(split,image.name,index,0)
        path=f.output/'g2_gating_analysis/gating_samples.tsv'
        analysis._write_delimited(path,list(samples[0]),samples)
        fields,rows=analysis._coefficient_rows(samples,self.result['selected_checkpoint']['sha256'])
        self.rows=rows
        per_sample=path.parent/'per_sample_gating.tsv';analysis._write_delimited(per_sample,fields,rows)
        stats=path.parent/'gate_statistics.csv';fields,stat_rows=analysis._statistics(rows);analysis._write_delimited(stats,fields,stat_rows,delimiter=',')
        summary_path=path.parent/'dynamic_gating_summary.json';summary=read_json(summary_path)
        summary['selected_sample_count']=6;summary['training_epoch_statistics']=gate_records[79]
        summary['gating_samples'].update(sha256=sha256_file(path),size_bytes=path.stat().st_size)
        write_json(summary_path,summary)
        manifest_path=Path(self.result['evidence']['analysis_manifest']);manifest=read_json(manifest_path)
        manifest.update(config_sha256=sha256_file(f.config),epoch_statistics_sha256=sha256_file(gate_path),plots_deferred=True,
                        static_dynamic_alpha=ALPHA,fusion_mode=self.result['fusion_mode'])
        manifest['files']={k:dict(v,sha256=sha256_file(v['path'])) for k,v in manifest['files'].items() if not k.endswith('_png')}
        for name,p in (('per_sample_gating_tsv',per_sample),('gate_statistics_csv',stats)):
            manifest['files'][name]={'path':str(p),'sha256':sha256_file(p)}
        write_json(manifest_path,manifest)
        self.result['evidence']['analysis_manifest_sha256']=sha256_file(manifest_path)
        self.result_path=f.output/profile.RESULT_FILENAME;write_json(self.result_path,self.result)
        self.lineage=mock.patch.object(recovery,'_lineage',return_value=dict(parent_branch=profile.EXPECTED_PARENT_BRANCH,parent_commit=profile.EXPECTED_PARENT_COMMIT,merge_base=profile.EXPECTED_PARENT_COMMIT))
        self.lineage.start();self.addCleanup(self.lineage.stop)

    def recover(self):
        f=self.fixture
        return profile.recover(f.config,f.output,f.console,f.records,f.experiments)

    def run_finish(self):
        f=self.fixture
        from tools.finish_g2_e_alpha_experiment import main
        with mock.patch.object(finalizer,'finalize',return_value=(self.result_path,self.result)):
            main(['--alpha',str(ALPHA),'--config-file',str(f.config),'--output-dir',str(f.output),
                  '--console-log',str(f.console),'--records-root',str(f.records),
                  '--experiments-path',str(f.experiments),'--archive-dir',str(f.root/'exports')])
        return read_json(f.output/'delivery_status.json')

    def test_registration_is_idempotent_and_preserves_history(self):
        f=self.fixture
        f.experiments.write_text('# Existing history\nKeep this paragraph.\n')
        first=self.recover();second=self.recover()
        self.assertTrue(first[2]);self.assertFalse(second[2])
        for relative in ('runs.csv','tables/main_results.csv'):
            self.assertEqual(len(read_rows(f.records/relative)),1)
        text=f.experiments.read_text(encoding='utf-8')
        self.assertIn('Keep this paragraph.',text)
        self.assertIn('mean alpha*w',text);self.assertIn('88.10000000',text)
        self.assertEqual(text.count('g2-e-alpha:'+first[0].name+':start'),1)
        self.assertEqual(read_json(first[0]/'run_manifest.json')['static_dynamic_alpha'],ALPHA)

    def test_invalid_metrics_or_raw_alpha_never_register(self):
        self.result['metrics']['map_percent']=99.
        write_json(self.result_path,self.result)
        with self.assertRaises((ValueError, recovery.G2RecoveryError)):self.recover()
        self.assertFalse((self.fixture.records/'runs.csv').exists())
        self.result['metrics']['map_percent']=88.1
        self.result['static_dynamic_alpha']=.5 if ALPHA==.1 else .1
        write_json(self.result_path,self.result)
        with self.assertRaises(ValueError):self.recover()
        self.assertFalse((self.fixture.records/'runs.csv').exists())

    def test_checkpoint_rejects_same_shape_other_alpha_or_training_commit(self):
        validate(self.checkpoint,self.configuration)
        wrong=self.configuration.clone();wrong.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA=.5 if ALPHA==.1 else .1
        with self.assertRaisesRegex(ValueError,'signature'):validate(self.checkpoint,wrong)
        path=Path(str(self.checkpoint)+'.metadata.json');meta=read_json(path);meta['training_commit']='a'*40;write_json(path,meta)
        with self.assertRaisesRegex(ValueError,'identity'):validate(self.checkpoint,self.configuration)

    def test_generic_evaluation_rejects_wrong_identity_before_loading_dataset(self):
        from tools import test as evaluate
        f=self.fixture
        metadata=Path(str(self.checkpoint)+'.metadata.json')
        value=read_json(metadata);value['training_commit']='a'*40;write_json(metadata,value)
        with mock.patch.object(evaluate,'cfg',cfg.clone()),mock.patch.object(evaluate.sys,'argv',['test.py','--config_file',str(f.config),'TEST.WEIGHT',str(self.checkpoint)]),mock.patch.object(evaluate,'make_data_loader') as loader:
            with self.assertRaisesRegex(ValueError,'identity'):evaluate.main()
        loader.assert_not_called()

    def test_smoke_preflight_binds_one_epoch_config_checkpoint_and_commit(self):
        from tools import verify_g2_e_alpha_smoke as smoke
        f=self.fixture
        output=f.root/'smoke';output.mkdir()
        configuration=self.configuration.clone()
        configuration.OUTPUT_DIR=str(output)
        configuration.SOLVER.MAX_EPOCHS=1;configuration.SOLVER.CHECKPOINT_PERIOD=1;configuration.SOLVER.EVAL_PERIOD=1
        resolved=output/'config_resolved.yml';resolved.write_text(serialize_cfg_node_yaml(configuration),encoding='utf-8')
        provenance=read_json(f.output/'reproducibility.json')
        provenance['configuration']['resolved_file_sha256']=sha256_file(resolved)
        write_json(output/'reproducibility.json',provenance)
        (output/'log.txt').write_text('EPOCH_EVIDENCE epoch=1 global_iteration=186 epoch_length=186\n')
        (output/'resnet50_checkpoint_186.pt').write_bytes(b'synthetic smoke fixture')
        row=json.loads((f.output/'validation_history.jsonl').read_text().splitlines()[0]);row.update(epoch=1,global_iteration=186)
        (output/'validation_history.jsonl').write_text(json.dumps(row)+'\n')
        seal(output)
        folder=output/'g2_e_smoke_analysis';folder.mkdir()
        rows=[dict(self.rows[0],stable_sample_key=format(i,'064x'),checkpoint_sha256=sha256_file(output/'resnet50_checkpoint_186.pt')) for i in range(32)]
        path=folder/'samples.tsv';analysis._write_delimited(path,list(rows[0]),rows)
        write_json(folder/(STEM+'_analysis_manifest.json'),dict(checkpoint_path=str(output/'resnet50_checkpoint_186.pt'),checkpoint_sha256=rows[0]['checkpoint_sha256'],files={'per_sample_gating_tsv':{'path':str(path),'sha256':sha256_file(path)}}))
        with mock.patch.object(smoke.subprocess,'check_output',return_value='f'*40):
            self.assertEqual(smoke.verify(ALPHA,f.config,output)['kind'],'smoke_only')
        with mock.patch.object(smoke.subprocess,'check_output',return_value='a'*40):
            with self.assertRaisesRegex(ValueError,'stale'):smoke.verify(ALPHA,f.config,output)

    def test_full_entrypoint_pack_failure_then_success_and_offline_replot(self):
        f=self.fixture
        with mock.patch.object(packager,'_resolve_samples',return_value=(self.mapping,self.dataset)), mock.patch.object(packager,'_diff',return_value='synthetic test fixture diff\n'):
            with mock.patch.object(packager,'_plot_distribution',side_effect=RuntimeError('injected plot failure')):
                with self.assertRaisesRegex(RuntimeError,'injected plot'):self.run_finish()
            self.assertEqual(len(read_rows(f.records/'tables/main_results.csv')),1)
            self.assertIn('plot_and_package_failed',f.experiments.read_text())
            outcome=self.run_finish()
            self.assertEqual(outcome['stage'],'complete')
            package=Path(outcome['package_dir'])
            self.assertTrue(Path(outcome['archive']).is_file())
            self.assertEqual(sha256_file(outcome['archive']),outcome['archive_sha256'])
            self.assertEqual(len(read_rows(f.records/'runs.csv')),1)
            for name in ('EXPERIMENTS.md','SHA256SUMS','gate_statistics.csv','selected_checkpoint.metadata.json','figures/k4_dominant_examples.png','figures/k6_dominant_examples.pdf'):
                self.assertTrue((package/name).is_file(),name)
            from tools.plot_g2_e_alpha_results import main
            main(['--package-dir',str(package),'--output-dir',str(f.root/'replot')])
            self.assertTrue((f.root/'replot/w_distribution.png').is_file())
            from tools.compare_g2_e_alpha_sweep import main as compare_main
            sources=f.root/'sources.json';write_json(sources,{'sources':[{'label':'E-test','alpha':ALPHA,'package_dir':str(package)}]})
            compare_main(['--sources',str(sources),'--output-dir',str(f.root/'comparison'),'--expected-count','6'])
            self.assertFalse(read_json(f.root/'comparison/comparison_manifest.json')['complete_alpha_sweep'])
            (package/'retrieval_metrics.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):archive_package(package,f.root/'other_export')

    def test_finalizer_reuses_provenance_commit_and_selected_epoch(self):
        f=self.fixture
        with mock.patch.object(finalizer,'_git',return_value=profile.EXPECTED_BRANCH), mock.patch.object(finalizer.subprocess,'check_call'):
            path,result=finalizer.finalize(f.config,f.output)
        self.assertEqual(result['selected_checkpoint']['epoch'],80)
        self.assertEqual(result['commit'],'f'*40)
        self.assertEqual(path,self.result_path)

    def test_new_finalization_registers_without_plotting(self):
        f=self.fixture
        self.result_path.unlink()
        manifest=Path(self.result['evidence']['analysis_manifest'])
        with mock.patch.object(finalizer,'_git',return_value=profile.EXPECTED_BRANCH),mock.patch.object(finalizer.subprocess,'check_call'),mock.patch.object(finalizer,'analyze',return_value=manifest) as analyze_call:
            path,result=finalizer.finalize(f.config,f.output)
        self.assertFalse(analyze_call.call_args.kwargs['render_plots'])
        self.assertEqual(analyze_call.call_args.kwargs['selected_epoch'],80)
        self.assertEqual(result['metrics']['map_percent'],88.1)
        self.assertEqual(path,self.result_path)
        self.recover()
        self.assertEqual(len(read_rows(f.records/'runs.csv')),1)

    def test_archive_failure_does_not_remove_registered_metrics(self):
        from tools import finish_g2_e_alpha_experiment as coordinator
        with mock.patch.object(packager,'package'),mock.patch.object(coordinator,'archive_package',side_effect=OSError('injected archive failure')):
            with self.assertRaisesRegex(OSError,'injected archive'):self.run_finish()
        f=self.fixture
        self.assertEqual(len(read_rows(f.records/'tables/main_results.csv')),1)
        self.assertIn('archive_failed',f.experiments.read_text())

    def test_two_alpha_identities_share_one_registry_without_overwrite(self):
        # Use a second independently constructed evidence tree, changing only its
        # alpha profile. Both checkpoint byte payloads deliberately have equal hashes.
        first=self.recover()
        other_alpha=.5 if ALPHA==.1 else .1
        other_id='C2-L03-MGDG-G2-E-SD-A{}-T0P5-S42'.format('05' if other_alpha==.5 else '01')
        other_branch='codex/g2-e-static-dynamic-alpha{}-tau0p5'.format(str(other_alpha).replace('.','p'))
        other=DeliveryIntegrationTest()
        original_profile=profile._configure_profile
        def configure_other():
            original_profile()
            recovery.EXPECTED_STATIC_DYNAMIC_ALPHA=other_alpha
        with mock.patch.dict(globals(),ALPHA=other_alpha),mock.patch.object(profile,'EXPECTED_BRANCH',other_branch),mock.patch.object(profile,'EXPERIMENT_ID',other_id),mock.patch.object(analysis,'ALPHA',other_alpha),mock.patch.object(profile,'_configure_profile',side_effect=configure_other):
            # Override the fixture source alpha while retaining all other baseline fields.
            source=yaml.safe_load(CONFIG.read_text());source['MODEL']['MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA']=other_alpha
            with tempfile.TemporaryDirectory() as temp:
                candidate=Path(temp)/'other.yml';candidate.write_text(yaml.safe_dump(source))
                with mock.patch.dict(globals(),CONFIG=candidate):
                    other.setUp()
                    try:
                        other.fixture.records=self.fixture.records;other.fixture.experiments=self.fixture.experiments
                        second=other.recover();other.recover()
                        self.assertNotEqual(first[0].name,second[0].name)
                        rows=read_rows(self.fixture.records/'tables/main_results.csv')
                        self.assertEqual(len(rows),2)
                        values={read_json(p/'run_manifest.json')['static_dynamic_alpha'] for p in (first[0],second[0])}
                        self.assertEqual(values,{.1,.5})
                    finally:other.doCleanups()

    def test_comparison_rejects_metadata_mismatch_without_intersection(self):
        from tools import compare_g2_e_alpha_sweep as comparison
        f=self.fixture
        source=dict(label='first',alpha=ALPHA,rows=self.rows,metrics=self.result['metrics'],provenance={})
        changed=json.loads(json.dumps(source));changed['label']='second';changed['alpha']=.3
        changed['rows'][0]['pid']='999'
        config=f.root/'sources.json'
        write_json(config,{'sources':[{'label':'first','package_dir':str(f.output)},{'label':'second','package_dir':str(f.output)}]})
        with mock.patch.object(comparison,'load_source',side_effect=[source,changed]):
            with self.assertRaisesRegex(ValueError,'no intersection'):comparison.compare(config,f.root/'bad_compare',6)
        self.assertFalse((f.root/'bad_compare').exists())

    def test_data_only_shared_analyzer_never_reads_missing_plots_and_uses_selected_epoch(self):
        shared=analysis.g2_analysis;f=self.fixture
        manifest=read_json(self.result['evidence']['analysis_manifest'])
        summary_path=Path(manifest['files']['dynamic_gating_summary_json']['path'])
        samples_path=Path(manifest['files']['test_gate_samples_tsv']['path'])
        summary=read_json(summary_path)
        with mock.patch.object(shared.torch,'load',return_value={}),mock.patch.object(shared,'_state_dict',return_value={}),mock.patch.object(shared,'_block_rows',return_value=([{'block':'g','norm':1.}],[])),mock.patch.object(shared,'generate_dynamic_gating_evidence',return_value=(summary_path,samples_path,summary)) as generate:
            with mock.patch.object(shared,'_plot_block_magnitudes',side_effect=AssertionError('plot called')),mock.patch.object(shared,'_plot_history',side_effect=AssertionError('plot called')),mock.patch.object(shared,'_plot_sample_weight_distribution',side_effect=AssertionError('plot called')):
                output=shared.analyze(f.config,self.checkpoint,f.root/'raw_only',f.output/'dynamic_gating_epoch_stats.jsonl',expected_gating_tau=.5,render_plots=False,selected_epoch=80)
        self.assertEqual(generate.call_args.args[3]['epoch'],80)
        self.assertFalse(any(k.endswith('_png') for k in read_json(output)['files']))


if __name__=='__main__':unittest.main()
