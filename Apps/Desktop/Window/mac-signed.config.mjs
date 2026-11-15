// The Mac app a person downloads: the package's build configuration, signed with the Developer ID Application
// certificate in this Mac's keychain, under the hardened runtime, notarized and stapled, in a disk image.
// build:mac stays the ad hoc folder for development; this is build:mac:signed.

import { readFileSync } from 'node:fs';

const base = JSON.parse( readFileSync( new URL( './package.json', import.meta.url ), 'utf8' ) ).build;

export default {
    ...base,
    mac: {
        target: [ { target: 'dmg', arch: [ 'arm64' ] } ],
        hardenedRuntime: true,
        entitlements: 'entitlements/mac.plist',
        entitlementsInherit: 'entitlements/mac.inherit.plist',
        notarize: true
    },
    dmg: {
        artifactName: 'Mila-Desktop-${version}-arm64.${ext}'
    }
};
