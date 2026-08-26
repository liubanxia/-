# Map Learning Layer

This module adds a lightweight 2D topology constraint for realtime screen-space prediction.

It does not render a 3D map and does not read game memory, world coordinates, packets, or hidden entity state.

Runtime inputs:

- last visible target point on screen
- target motion vector
- coarse audio direction
- floor relation: above / same / below / unknown
- current map ID
- current nearest learned topology node

The map profile is reduced to nodes and edges such as corridors, doors, corners, stairs, floor links, choke points, and common routes. At runtime the predictor ranks valid outgoing routes and converts them into a few blue screen-space prediction points.

Supported map IDs are seeded for Zero Dam, Space City, Layali Grove, Brakkesh, and Tide Prison. Detailed topology data is intentionally stored as replaceable JSON and should be generated from lawful public or user-provided map material rather than copied from proprietary game assets.

Realtime media remains zero-retention. Map knowledge is static configuration, not captured gameplay history.
