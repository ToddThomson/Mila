import { test } from 'node:test';
import assert from 'node:assert/strict';

import { areas, settings, isReady, isShown, searchSettings, settingsInArea } from '../views/settings-table.js';

function pageState( session )
{
    return { session, installedModels: [], publishedModels: null, progress: null, modelsError: null };
}

function loadedSession( change )
{
    const session = {
        model: 'gemma-4-12b-it',
        context_length: 32768,
        context_automatic: true,
        thinking_capable: true,
        thinking: 'on',
        effort: null,
        effort_levels: null,
        detail: 'thoughts',
        sampling: {
            temperature: { value: 1, is_model_default: true },
            top_k: { value: 64, is_model_default: true },
            top_p: { value: 0.95, is_model_default: true }
        },
        web_search: false,
        run_commands: false,
        run_commands_problem: null,
        program_folders: []
    };

    return { ...session, ...change };
}

function identifiers( found )
{
    return found.map( ( setting ) => setting.id );
}

function everySettingIsDeclaredOnceInAnArea()
{
    const seen = new Set();
    const areaIdentifiers = new Set( areas.map( ( area ) => area.id ) );

    for ( const setting of settings )
    {
        assert.ok( !seen.has( setting.id ), setting.id + ' is declared twice' );
        assert.ok( areaIdentifiers.has( setting.area ), setting.id + ' names no area' );
        seen.add( setting.id );
    }

    for ( const area of areas )
        assert.ok( settings.some( ( setting ) => setting.area === area.id ), area.label + ' has no setting' );
}

test( 'every setting is declared once, in an area, and every area has one', everySettingIsDeclaredOnceInAnArea );

function aSearchFindsASettingByItsLabelItsAreaOrItsWords()
{
    const state = pageState( loadedSession( {} ) );

    assert.deepEqual( identifiers( searchSettings( 'temperature', state ) ), [ 'temperature' ] );
    assert.deepEqual( identifiers( searchSettings( 'TAVILY', state ) ), [ 'search_key' ] );
    assert.deepEqual( identifiers( searchSettings( 'repetition', state ) ), [] );
    assert.ok( identifiers( searchSettings( 'install', state ) ).includes( 'available' ) );
    assert.deepEqual( identifiers( searchSettings( 'sampling top', state ) ), [ 'top_k', 'top_p' ] );
    assert.deepEqual( identifiers( searchSettings( '   ', state ) ), [] );
}

test( 'a search finds a setting by its label, its area or its words', aSearchFindsASettingByItsLabelItsAreaOrItsWords );

function aSettingTheModelDoesNotHaveIsLeftOut()
{
    const withoutThinking = pageState( loadedSession( { thinking_capable: false } ) );
    const withEffort = pageState( loadedSession( { effort_levels: [ 'low', 'medium', 'high' ], effort: 'medium' } ) );

    assert.deepEqual( identifiers( settingsInArea( 'responses', withoutThinking ) ), [ 'context' ] );
    assert.deepEqual( identifiers( settingsInArea( 'responses', withEffort ) ), [ 'thinking', 'effort', 'context' ] );
    assert.deepEqual( identifiers( settingsInArea( 'sampling', withoutThinking ) ), [ 'temperature', 'top_k', 'top_p', 'seed' ] );
}

test( 'a setting the model does not have is left out', aSettingTheModelDoesNotHaveIsLeftOut );

function withNoModelTheModelsSettingsAreListedWithNoControl()
{
    const noModel = pageState( loadedSession( { model: null } ) );
    const starting = pageState( null );
    const temperature = settings.find( ( setting ) => setting.id === 'temperature' );
    const searchKey = settings.find( ( setting ) => setting.id === 'search_key' );
    const installed = settings.find( ( setting ) => setting.id === 'installed' );

    assert.ok( isShown( temperature, noModel ) );
    assert.ok( !isReady( temperature, noModel ) );
    assert.ok( isReady( searchKey, noModel ) );
    assert.ok( !isReady( searchKey, starting ) );
    assert.ok( isReady( installed, starting ) );
    assert.deepEqual( identifiers( searchSettings( 'thinking', starting ) ), [ 'thinking', 'effort', 'detail' ] );
}

test( 'with no model, the model\'s settings are listed with no control', withNoModelTheModelsSettingsAreListedWithNoControl );
