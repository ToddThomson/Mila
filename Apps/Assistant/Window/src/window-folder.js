// Where the window keeps Electron's own data: the Assistant area of the person's Mila folder, in a folder of its own
// beside the workspace's (MilaAssistant.md 6.2). Named here so the product's name does not decide it, as Electron's
// default would. The same Mila folder as the session's `resolveMilaFolder()`.

import path from 'node:path';

/**
 * @param platform process.platform.
 * @param environment process.env.
 * @param home The person's home folder.
 */
export function windowDataFolder( platform, environment, home )
{
    if ( platform === 'win32' )
    {
        const localAppData = environment.LOCALAPPDATA;
        const root = localAppData !== undefined && localAppData.length > 0 ? localAppData : path.win32.join( home, 'AppData', 'Local' );

        return path.win32.join( root, 'Mila', 'Assistant', 'Window' );
    }

    if ( platform === 'darwin' )
        return path.posix.join( home, 'Library', 'Application Support', 'Mila', 'Assistant', 'Window' );

    const dataHome = environment.XDG_DATA_HOME;
    const root = dataHome !== undefined && dataHome.length > 0 ? dataHome : path.posix.join( home, '.local', 'share' );

    return path.posix.join( root, 'mila', 'assistant', 'window' );
}
