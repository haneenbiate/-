"""Package independent, prediction-blinded private review files deterministically."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from nasn.task05 import HUMAN, csv, fresh, registry, save
from nasn.preflight import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-package',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path(args.review_package)
    registry(source)
    protocol=Path('docs/task05_review_protocol_ar.md').read_bytes()
    output=fresh(args.output)
    archive_hashes={}
    for reviewer in ['a','b']:
        frame=csv(source/f'reviewer_{reviewer}.csv')
        require(frame[HUMAN].eq('').all().all(),'Review package must contain only blank assignments')
        members={f'reviewer_{reviewer}.csv':(source/f'reviewer_{reviewer}.csv').read_bytes(),
                 f'reviewer_{reviewer}.html':(source/f'reviewer_{reviewer}.html').read_bytes(),
                 'review_protocol_ar.md':protocol}
        checks={name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
        members['package_checksums.json']=(json.dumps(checks,indent=2)+'\n').encode()
        file=output/f'reviewer_{reviewer}.zip'
        with zipfile.ZipFile(file,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for name,data in sorted(members.items()):
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o600<<16
                bundle.writestr(info,data)
        archive_hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    save(output/'archive_sha256.json',archive_hashes)
    print(json.dumps(archive_hashes,indent=2))


if __name__=='__main__':main()
