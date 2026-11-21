import { test } from 'node:test';
import assert from 'node:assert/strict';
import { windowDataFolder } from '../src/window-folder.js';

test( 'the window keeps its data in the Assistant area of the Mila folder on each platform', () =>
{
    assert.equal( windowDataFolder( 'win32', { LOCALAPPDATA: 'C:\\Users\\a\\AppData\\Local' }, 'C:\\Users\\a' ),
        'C:\\Users\\a\\AppData\\Local\\Mila\\Assistant\\Window' );
    assert.equal( windowDataFolder( 'darwin', {}, '/Users/a' ), '/Users/a/Library/Application Support/Mila/Assistant/Window' );
    assert.equal( windowDataFolder( 'linux', { XDG_DATA_HOME: '/data' }, '/home/a' ), '/data/mila/assistant/window' );
    assert.equal( windowDataFolder( 'linux', {}, '/home/a' ), '/home/a/.local/share/mila/assistant/window' );
} );

test( 'an empty variable reads as absent, as the session reads it', () =>
{
    assert.equal( windowDataFolder( 'win32', { LOCALAPPDATA: '' }, 'C:\\Users\\a' ), 'C:\\Users\\a\\AppData\\Local\\Mila\\Assistant\\Window' );
    assert.equal( windowDataFolder( 'linux', { XDG_DATA_HOME: '' }, '/home/a' ), '/home/a/.local/share/mila/assistant/window' );
} );
